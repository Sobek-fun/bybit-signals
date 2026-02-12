import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import clickhouse_connect
import numpy as np
import pandas as pd
import pandas_ta as ta
from catboost import CatBoostClassifier

from pump_end.features.feature_builder import PumpFeatureBuilder
from pump_end.features.params import PumpParams, DEFAULT_PUMP_PARAMS
from pump_end.infra.clickhouse import DataLoader
from pump_end.ml.feature_schema import prune_feature_columns
from pump_end.ml.predict import extract_signals


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def load_run_config(model_dir: Path) -> dict:
    config_path = model_dir / "run_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def load_threshold_config(model_dir: Path) -> dict:
    threshold_path = model_dir / "best_threshold.json"
    with open(threshold_path, 'r') as f:
        return json.load(f)


def load_model(model_dir: Path) -> CatBoostClassifier:
    model_path = model_dir / "catboost_model.cbm"
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def get_symbols(client, start_dt: datetime, end_dt: datetime) -> list:
    query = """
            SELECT DISTINCT symbol
            FROM bybit.candles
            WHERE open_time >= %(start)s
              AND open_time < %(end)s
              AND interval = 1
            ORDER BY symbol \
            """
    result = client.query(query, parameters={
        "start": start_dt,
        "end": end_dt
    })
    return [row[0] for row in result.result_rows]


def detect_candidates(df: pd.DataFrame, params: PumpParams) -> list:
    df = df.copy()

    df['vol_median'] = df['volume'].rolling(window=params.volume_median_window).median()
    df['vol_ratio'] = df['volume'] / df['vol_median']

    df.ta.rsi(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df = df.rename(columns={
        'RSI_14': 'rsi_14',
        'MFI_14': 'mfi_14',
        'MACDh_12_26_9': 'macdh_12_26_9',
        'MACD_12_26_9': 'macd_line',
        'MACDs_12_26_9': 'macd_signal'
    })

    rsi = df['rsi_14']
    mfi = df['mfi_14']
    macdh = df['macdh_12_26_9']
    macd_line = df['macd_line']

    df['vol_ratio_max'] = df['vol_ratio'].rolling(window=params.peak_window).max()
    df['rsi_max'] = rsi.rolling(window=params.peak_window).max()
    df['macdh_max'] = macdh.rolling(window=params.peak_window).max()
    df['high_max'] = df['high'].rolling(window=params.peak_window).max()

    min_price = df[['low', 'open']].min(axis=1)
    local_min = min_price.rolling(window=params.runup_window).min()
    runup = (df['high'] / local_min) - 1
    runup_met = ((local_min > 0) & (runup >= params.runup_threshold)).fillna(False)

    vol_spike_cond = df['vol_ratio'] >= params.vol_ratio_spike
    vol_spike_recent = vol_spike_cond.rolling(window=params.context_window).sum().fillna(0) > 0

    rsi_corridor = rsi.rolling(window=params.corridor_window).quantile(params.corridor_quantile)
    mfi_corridor = mfi.rolling(window=params.corridor_window).quantile(params.corridor_quantile)

    rsi_hot = rsi.notna() & rsi_corridor.notna() & (rsi >= np.maximum(params.rsi_hot, rsi_corridor))
    mfi_hot = mfi.notna() & mfi_corridor.notna() & (mfi >= np.maximum(params.mfi_hot, mfi_corridor))
    osc_hot_recent = (rsi_hot | mfi_hot).rolling(window=params.context_window).sum().fillna(0) > 0

    macd_pos_recent = (macdh.notna() & (macdh > 0)).rolling(window=params.context_window).sum().fillna(0) > 0

    pump_ctx = runup_met & vol_spike_recent & osc_hot_recent & macd_pos_recent

    high_max = df['high_max']
    near_peak = high_max.notna() & (high_max > 0) & (df['high'] >= high_max * (1 - params.peak_tol))

    n = len(df)
    open_p = df['open'].values
    high_p = df['high'].values
    low_p = df['low'].values
    close_p = df['close'].values

    candle_range_arr = high_p - low_p
    range_pos_arr = candle_range_arr > 0

    close_pos = np.zeros(n, dtype=float)
    np.divide(close_p - low_p, candle_range_arr, out=close_pos, where=range_pos_arr)

    max_oc = np.maximum(open_p, close_p)
    upper_wick = high_p - max_oc

    wick_ratio = np.zeros(n, dtype=float)
    np.divide(upper_wick, candle_range_arr, out=wick_ratio, where=range_pos_arr)

    body_size = np.abs(close_p - open_p)
    body_ratio = np.zeros(n, dtype=float)
    np.divide(body_size, candle_range_arr, out=body_ratio, where=range_pos_arr)

    bearish = close_p < open_p

    blowoff_exhaustion = (
            (close_pos <= params.close_pos_low) |
            (bearish & (close_pos <= 0.45)) |
            ((wick_ratio >= params.wick_blowoff) & (body_ratio <= params.body_blowoff))
    )

    osc_extreme = rsi.notna() & mfi.notna() & (rsi >= params.rsi_extreme) & (mfi >= params.mfi_extreme)
    predump_mask = osc_extreme & (close_pos >= params.close_pos_high)

    vol_ratio_max = df['vol_ratio_max']
    vol_ratio = df['vol_ratio']
    vol_fade = vol_ratio_max.notna() & (vol_ratio_max > 0) & (vol_ratio <= vol_ratio_max * params.vol_fade_ratio)

    wick_high_mask = wick_ratio >= params.wick_high
    wick_low_mask = (wick_ratio >= params.wick_low) & (~wick_high_mask)

    rsi_max = df['rsi_max']
    macdh_max = df['macdh_max']
    rsi_fade = rsi_max.notna() & (rsi_max > 0) & (rsi <= rsi_max * params.rsi_fade_ratio)
    macd_fade = macdh_max.notna() & macdh.notna() & (macdh_max > 0) & (macdh <= macdh_max * params.macd_fade_ratio)

    predump_peak = (
            predump_mask &
            (
                    (wick_high_mask & vol_fade) |
                    (wick_low_mask & vol_fade & (rsi_fade | macd_fade))
            )
    ).fillna(False)

    pump_ctx_arr = pump_ctx.to_numpy(dtype=bool, copy=False)
    near_peak_arr = near_peak.to_numpy(dtype=bool, copy=False)
    blowoff_arr = np.array(blowoff_exhaustion, dtype=bool)
    predump_arr = predump_peak.to_numpy(dtype=bool, copy=False)

    strong_cond = pump_ctx_arr & near_peak_arr & (blowoff_arr | predump_arr)

    macd_turn_down = (
            macd_line.notna() &
            macd_line.shift(1).notna() &
            (macd_line < macd_line.shift(1))
    ).to_numpy(dtype=bool, copy=False)

    candidates = []
    pending = False
    last_signal_idx = None

    skip_initial = max(
        params.volume_median_window,
        params.runup_window,
        params.corridor_window,
        params.context_window,
        params.peak_window
    )

    for i in range(skip_initial, n):
        if strong_cond[i] and not pending:
            pending = True

        if pending and macd_turn_down[i]:
            if last_signal_idx is None or i - last_signal_idx >= params.cooldown_bars:
                candidates.append(df.index[i])
                last_signal_idx = i
            pending = False

    return candidates


def build_points_around_candidates(
        candidates: list,
        symbol: str,
        neg_before: int,
        neg_after: int,
        pos_offsets: list
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()

    all_offsets = list(range(-neg_before, 0)) + pos_offsets + list(
        range(max(pos_offsets) + 1, max(pos_offsets) + neg_after + 1)
    )
    all_offsets = sorted(set(all_offsets))

    rows = []
    for event_time in candidates:
        event_id = f"{symbol}|{event_time.strftime('%Y%m%d_%H%M%S')}"
        for offset in all_offsets:
            open_time = event_time + timedelta(minutes=offset * 15)
            rows.append({
                'event_id': event_id,
                'symbol': symbol,
                'open_time': open_time,
                'offset': offset,
                'y': 1 if offset in pos_offsets else 0,
                'pump_la_type': 'A',
                'runup_pct': 0
            })

    return pd.DataFrame(rows)


def process_symbol_chunk(args_tuple):
    (
        worker_id,
        symbols_chunk,
        ch_dsn,
        start_dt,
        end_dt,
        model_path,
        feature_columns,
        window_bars,
        warmup_bars,
        feature_set,
        threshold,
        min_pending_bars,
        drop_delta,
        neg_before,
        neg_after,
        pos_offsets,
        params_dict
    ) = args_tuple

    params = PumpParams(**params_dict) if params_dict else DEFAULT_PUMP_PARAMS

    loader = DataLoader(ch_dsn)
    builder = PumpFeatureBuilder(
        ch_dsn=ch_dsn,
        window_bars=window_bars,
        warmup_bars=warmup_bars,
        feature_set=feature_set,
        params=params
    )

    model = CatBoostClassifier()
    model.load_model(model_path)

    csv_filename = f"signals_part_{worker_id}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['symbol', 'timestamp'])

    total_signals = 0
    symbols_processed = 0
    candidates_found = 0

    buffer_bars = warmup_bars + window_bars + params.liquidity_window_bars + 21

    for symbol in symbols_chunk:
        try:
            query_start = start_dt - timedelta(minutes=buffer_bars * 15)
            df = loader.load_candles_range(symbol, query_start, end_dt)

            if df.empty or len(df) < buffer_bars:
                symbols_processed += 1
                continue

            candidates = detect_candidates(df, params)
            candidates = [c for c in candidates if start_dt <= c < end_dt]

            if not candidates:
                symbols_processed += 1
                continue

            candidates_found += len(candidates)

            points_df = build_points_around_candidates(
                candidates, symbol, neg_before, neg_after, pos_offsets
            )

            if points_df.empty:
                symbols_processed += 1
                continue

            points_df = points_df.sort_values('y', ascending=False)
            points_df = points_df.drop_duplicates(subset=['symbol', 'open_time'], keep='first')
            points_df = points_df.sort_values(['event_id', 'offset']).reset_index(drop=True)

            feature_input = points_df[['symbol', 'open_time']].copy()
            feature_input = feature_input.rename(columns={'open_time': 'event_open_time'})
            feature_input['pump_la_type'] = 'A'
            feature_input['runup_pct'] = 0

            features_df = builder.build(feature_input, max_workers=1)

            if features_df.empty:
                symbols_processed += 1
                continue

            features_df = features_df.merge(
                points_df[['symbol', 'open_time', 'event_id', 'offset', 'y']],
                on=['symbol', 'open_time'],
                how='inner'
            )

            if features_df.empty:
                symbols_processed += 1
                continue

            available_features = [f for f in feature_columns if f in features_df.columns]
            if len(available_features) < len(feature_columns):
                symbols_processed += 1
                continue

            X = features_df[feature_columns]
            p_end = model.predict_proba(X)[:, 1]

            predictions_df = features_df[['event_id', 'symbol', 'open_time', 'offset', 'y']].copy()
            predictions_df['p_end'] = p_end
            predictions_df['split'] = 'test'

            signals_df = extract_signals(
                predictions_df,
                threshold,
                min_pending_bars=min_pending_bars,
                drop_delta=drop_delta
            )

            for _, row in signals_df.iterrows():
                timestamp_str = pd.to_datetime(row['open_time']).strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([row['symbol'], timestamp_str])
                total_signals += 1

            symbols_processed += 1

        except Exception as e:
            symbols_processed += 1
            continue

    csv_file.close()

    return worker_id, symbols_processed, candidates_found, total_signals, csv_filename


def main():
    parser = argparse.ArgumentParser(description="Export pump end signals from trained model")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD HH:MM:SS), exclusive"
    )
    parser.add_argument(
        "--clickhouse-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN (e.g., http://user:pass@host:port/database)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model artifacts directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pump_end_signals.csv",
        help="Output CSV file path (default: pump_end_signals.csv)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M:%S')
    model_dir = Path(args.model_dir)

    log("INFO", "EXPORT", f"loading artifacts from {model_dir}")
    run_config = load_run_config(model_dir)
    threshold_config = load_threshold_config(model_dir)

    window_bars = run_config.get('window_bars', 30)
    warmup_bars = run_config.get('warmup_bars', 150)
    feature_set = run_config.get('feature_set', 'base')
    do_prune = run_config.get('prune_features', False)
    neg_before = run_config.get('neg_before', 20)
    neg_after = run_config.get('neg_after', 0)
    pos_offsets_str = run_config.get('pos_offsets', '0')
    pos_offsets = [int(x.strip()) for x in pos_offsets_str.split(',')]

    threshold = threshold_config['threshold']
    min_pending_bars = threshold_config.get('min_pending_bars', 1)
    drop_delta = threshold_config.get('drop_delta', 0.0)

    log("INFO", "EXPORT",
        f"config: window_bars={window_bars} warmup_bars={warmup_bars} feature_set={feature_set} prune={do_prune}")
    log("INFO", "EXPORT",
        f"threshold={threshold} min_pending_bars={min_pending_bars} drop_delta={drop_delta}")
    log("INFO", "EXPORT",
        f"neg_before={neg_before} neg_after={neg_after} pos_offsets={pos_offsets}")

    model = load_model(model_dir)
    feature_columns = list(model.feature_names_)

    if do_prune:
        original_count = len(feature_columns)
        feature_columns = prune_feature_columns(feature_columns)
        log("INFO", "EXPORT", f"pruned features: {original_count} -> {len(feature_columns)}")

    parsed = urlparse(args.clickhouse_dsn)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8123
    username = parsed.username or "default"
    password = parsed.password or ""
    database = parsed.path.lstrip("/") if parsed.path else "default"
    secure = parsed.scheme == "https"

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        secure=secure
    )

    log("INFO", "EXPORT", f"fetching symbols from {args.start_date} to {args.end_date}")
    symbols = get_symbols(client, start_dt, end_dt)

    if not symbols:
        log("WARN", "EXPORT", "no symbols found")
        return

    log("INFO", "EXPORT", f"found {len(symbols)} symbols")

    num_workers = min(args.workers, len(symbols))
    chunk_size = len(symbols) // num_workers
    remainder = len(symbols) % num_workers

    chunks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(symbols[start_idx:end_idx])
        start_idx = end_idx

    params_dict = {
        'runup_window': DEFAULT_PUMP_PARAMS.runup_window,
        'runup_threshold': DEFAULT_PUMP_PARAMS.runup_threshold,
        'context_window': DEFAULT_PUMP_PARAMS.context_window,
        'peak_window': DEFAULT_PUMP_PARAMS.peak_window,
        'peak_tol': DEFAULT_PUMP_PARAMS.peak_tol,
        'volume_median_window': DEFAULT_PUMP_PARAMS.volume_median_window,
        'vol_ratio_spike': DEFAULT_PUMP_PARAMS.vol_ratio_spike,
        'vol_fade_ratio': DEFAULT_PUMP_PARAMS.vol_fade_ratio,
        'corridor_window': DEFAULT_PUMP_PARAMS.corridor_window,
        'corridor_quantile': DEFAULT_PUMP_PARAMS.corridor_quantile,
        'rsi_hot': DEFAULT_PUMP_PARAMS.rsi_hot,
        'mfi_hot': DEFAULT_PUMP_PARAMS.mfi_hot,
        'rsi_extreme': DEFAULT_PUMP_PARAMS.rsi_extreme,
        'mfi_extreme': DEFAULT_PUMP_PARAMS.mfi_extreme,
        'rsi_fade_ratio': DEFAULT_PUMP_PARAMS.rsi_fade_ratio,
        'macd_fade_ratio': DEFAULT_PUMP_PARAMS.macd_fade_ratio,
        'wick_high': DEFAULT_PUMP_PARAMS.wick_high,
        'wick_low': DEFAULT_PUMP_PARAMS.wick_low,
        'close_pos_high': DEFAULT_PUMP_PARAMS.close_pos_high,
        'close_pos_low': DEFAULT_PUMP_PARAMS.close_pos_low,
        'wick_blowoff': DEFAULT_PUMP_PARAMS.wick_blowoff,
        'body_blowoff': DEFAULT_PUMP_PARAMS.body_blowoff,
        'cooldown_bars': DEFAULT_PUMP_PARAMS.cooldown_bars,
        'liquidity_window_bars': DEFAULT_PUMP_PARAMS.liquidity_window_bars,
        'eqh_min_touches': DEFAULT_PUMP_PARAMS.eqh_min_touches,
        'eqh_base_tol': DEFAULT_PUMP_PARAMS.eqh_base_tol,
        'eqh_atr_factor': DEFAULT_PUMP_PARAMS.eqh_atr_factor,
    }

    model_path = str(model_dir / "catboost_model.cbm")

    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append((
            i + 1,
            chunk,
            args.clickhouse_dsn,
            start_dt,
            end_dt,
            model_path,
            feature_columns,
            window_bars,
            warmup_bars,
            feature_set,
            threshold,
            min_pending_bars,
            drop_delta,
            neg_before,
            neg_after,
            pos_offsets,
            params_dict
        ))

    log("INFO", "EXPORT", f"starting {num_workers} workers")

    part_files = []
    total_processed = 0
    total_candidates = 0
    total_signals = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_symbol_chunk, task): task[0] for task in tasks}

        for future in as_completed(futures):
            worker_id, processed, candidates, signals, csv_file = future.result()
            part_files.append(csv_file)
            total_processed += processed
            total_candidates += candidates
            total_signals += signals
            log("INFO", "EXPORT",
                f"worker={worker_id} done symbols={processed} candidates={candidates} signals={signals}")

    with open(args.output, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['symbol', 'timestamp'])

        for part_csv in part_files:
            if os.path.exists(part_csv):
                with open(part_csv, 'r') as infile:
                    reader = csv.reader(infile)
                    next(reader)
                    for row in reader:
                        csv_writer.writerow(row)
                os.remove(part_csv)

    log("INFO", "EXPORT",
        f"done symbols={total_processed} candidates={total_candidates} signals={total_signals} output={args.output}")


if __name__ == "__main__":
    main()
