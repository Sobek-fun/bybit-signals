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

from pump_end_threshold.features.feature_builder import PumpFeatureBuilder
from pump_end_threshold.features.params import PumpParams, DEFAULT_PUMP_PARAMS
from pump_end_threshold.infra.clickhouse import DataLoader
from pump_end_threshold.ml.feature_schema import prune_feature_columns
from pump_end_threshold.ml.predict import extract_signals_verbose


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
    from pump_end_threshold.tools.pump_start_detection.detector import PumpDetector

    df = df.copy()

    df.ta.rsi(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    detector = PumpDetector(params)
    result = detector.detect(df)

    candidates = result[result['pump_signal'] == 'strong_pump'].index.tolist()
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
        params_dict,
        abstain_margin,
        temp_dir
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

    all_signals = []
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

            signals_df = extract_signals_verbose(
                predictions_df,
                threshold,
                min_pending_bars=min_pending_bars,
                drop_delta=drop_delta,
                abstain_margin=abstain_margin
            )

            if not signals_df.empty:
                all_signals.append(signals_df)
                total_signals += len(signals_df)

            symbols_processed += 1

        except Exception:
            symbols_processed += 1
            continue

    part_file = str(Path(temp_dir) / f"signals_part_{worker_id}.parquet")
    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        combined.to_parquet(part_file, index=False)
    else:
        pd.DataFrame().to_parquet(part_file, index=False)

    return worker_id, symbols_processed, candidates_found, total_signals, part_file


def run_guard_stage(
        raw_signals_df: pd.DataFrame,
        guard_model_dir: Path,
        ch_dsn: str,
        run_dir: Path = None,
        guard_debug_output: str = None,
        blocked_signals_output: str = None,
        accepted_signals_output: str = None,
) -> pd.DataFrame:
    from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
    from pump_end_threshold.ml.regime_policy import RegimePolicy
    from pump_end_threshold.infra.clickhouse import get_liquid_universe

    guard_model_path = guard_model_dir / "regime_guard_model.cbm"
    guard_model = CatBoostClassifier()
    guard_model.load_model(str(guard_model_path))

    policy_path = guard_model_dir / "best_policy_params.json"
    with open(policy_path, 'r') as f:
        policy_params = json.load(f)

    feature_cols_path = guard_model_dir / "feature_columns.json"
    with open(feature_cols_path, 'r') as f:
        guard_feature_columns = json.load(f)

    signals = raw_signals_df.sort_values('open_time').reset_index(drop=True)
    t_min = signals['open_time'].min()
    t_max = signals['open_time'].max()

    # Try to load liquid_universe from saved config
    liquid_universe_path = guard_model_dir / "liquid_universe.json"
    if liquid_universe_path.exists():
        log("INFO", "GUARD", f"loading liquid_universe from {liquid_universe_path}")
        with open(liquid_universe_path, 'r') as f:
            liquid_universe = json.load(f)
    else:
        log("WARN", "GUARD", "liquid_universe.json not found, computing from scratch")
        liquid_universe = get_liquid_universe(
            ch_dsn, t_min - timedelta(days=7), t_max, top_n=120
        )

    # Try to load regime builder config
    regime_config_path = guard_model_dir / "regime_builder_config.json"
    if regime_config_path.exists():
        log("INFO", "GUARD", f"loading regime builder config from {regime_config_path}")
        with open(regime_config_path, 'r') as f:
            regime_config = json.load(f)
        top_n = regime_config.get('top_n_universe', 120)
    else:
        log("WARN", "GUARD", "regime_builder_config.json not found, using defaults")
        top_n = 120

    builder = RegimeFeatureBuilder(
        ch_dsn=ch_dsn,
        liquid_universe=liquid_universe,
        top_n=top_n,
    )

    log("INFO", "GUARD", "building regime features in batch mode")
    guard_features = builder.build_batch(signals, batch_size=100)

    if guard_features.empty:
        log("WARN", "GUARD", "no features built")
        return raw_signals_df

    available = [c for c in guard_feature_columns if c in guard_features.columns]
    if len(available) < len(guard_feature_columns):
        missing = set(guard_feature_columns) - set(available)
        log("WARN", "GUARD", f"missing {len(missing)} feature columns, filling with NaN: {list(missing)[:10]}")
        for col in missing:
            guard_features[col] = np.nan

    X_guard = guard_features[guard_feature_columns]
    p_bad = guard_model.predict_proba(X_guard)[:, 1]

    signals['p_bad'] = p_bad

    if guard_debug_output:
        signals.to_parquet(guard_debug_output, index=False)
    elif run_dir:
        guard_scored_path = run_dir / "guard_scored_signals.parquet"
        signals.to_parquet(guard_scored_path, index=False)

    policy = RegimePolicy(**policy_params)
    result = policy.apply(signals, p_bad_col='p_bad')

    accepted = result[~result['blocked_by_policy']].copy()
    blocked = result[result['blocked_by_policy']].copy()

    if accepted_signals_output:
        accepted.to_parquet(accepted_signals_output, index=False)
    elif run_dir:
        accepted.to_parquet(run_dir / "accepted_signals.parquet", index=False)

    if blocked_signals_output:
        blocked.to_parquet(blocked_signals_output, index=False)
    elif run_dir:
        blocked.to_parquet(run_dir / "blocked_signals.parquet", index=False)

    n_blocked = result['blocked_by_policy'].sum()
    log("INFO", "GUARD", f"guard applied: {len(result)} raw -> {len(accepted)} accepted ({n_blocked} blocked)")

    return accepted


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
        help="Path to detector model artifacts directory"
    )
    parser.add_argument(
        "--guard-model-dir",
        type=str,
        default=None,
        help="Path to regime guard model directory (optional, enables guard stage)"
    )
    parser.add_argument(
        "--skip-guard",
        action="store_true",
        help="Skip guard stage even if guard model exists in run-dir"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for all artifacts (if not provided, uses individual output paths)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: RUN_DIR/final_signals.csv or pump_end_signals.csv)"
    )
    parser.add_argument(
        "--raw-signals-output",
        type=str,
        default=None,
        help="Optional: save raw detector signals to this parquet path (default: RUN_DIR/raw_detector_signals.parquet if --run-dir set)"
    )
    parser.add_argument(
        "--guard-debug-output",
        type=str,
        default=None,
        help="Optional: save guard scored signals to this parquet path"
    )
    parser.add_argument(
        "--blocked-signals-output",
        type=str,
        default=None,
        help="Optional: save blocked signals to this parquet path"
    )
    parser.add_argument(
        "--accepted-signals-output",
        type=str,
        default=None,
        help="Optional: save accepted signals to this parquet path"
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

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = run_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        if not args.output:
            args.output = str(run_dir / "final_signals.csv")
        if not args.raw_signals_output:
            args.raw_signals_output = str(run_dir / "raw_detector_signals.parquet")
    else:
        run_dir = None
        temp_dir = Path(".")
        if not args.output:
            args.output = "pump_end_signals.csv"

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
    abstain_margin = threshold_config.get('abstain_margin', 0.0)

    log("INFO", "EXPORT",
        f"config: window_bars={window_bars} warmup_bars={warmup_bars} feature_set={feature_set} prune={do_prune}")
    log("INFO", "EXPORT",
        f"threshold={threshold} min_pending_bars={min_pending_bars} drop_delta={drop_delta} abstain_margin={abstain_margin}")
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
            params_dict,
            abstain_margin,
            str(temp_dir)
        ))

    log("INFO", "EXPORT", f"stage A: starting {num_workers} detector workers")

    part_files = []
    total_processed = 0
    total_candidates = 0
    total_signals = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_symbol_chunk, task): task[0] for task in tasks}

        for future in as_completed(futures):
            worker_id, processed, candidates, signals, part_file = future.result()
            part_files.append(part_file)
            total_processed += processed
            total_candidates += candidates
            total_signals += signals
            log("INFO", "EXPORT",
                f"worker={worker_id} done symbols={processed} candidates={candidates} signals={signals}")

    log("INFO", "EXPORT", f"stage A done: {total_signals} raw detector signals from {total_candidates} candidates")

    all_raw = []
    for pf in part_files:
        if os.path.exists(pf):
            part_df = pd.read_parquet(pf)
            if not part_df.empty:
                all_raw.append(part_df)
            os.remove(pf)

    if not all_raw:
        log("WARN", "EXPORT", "no raw signals produced")
        with open(args.output, 'w', newline='') as outfile:
            csv.writer(outfile).writerow(['symbol', 'timestamp'])
        return

    raw_signals = pd.concat(all_raw, ignore_index=True)
    raw_signals = raw_signals.sort_values('open_time').reset_index(drop=True)

    log("INFO", "EXPORT", f"combined {len(raw_signals)} raw detector signals")

    if args.raw_signals_output:
        raw_signals.to_parquet(args.raw_signals_output, index=False)
        log("INFO", "EXPORT", f"raw signals saved to {args.raw_signals_output}")

    if args.skip_guard:
        log("INFO", "EXPORT", "stage B: skipping guard stage as requested")
        final_signals = raw_signals
    elif args.guard_model_dir:
        guard_dir = Path(args.guard_model_dir)
        log("INFO", "EXPORT", f"stage B: applying regime guard from {guard_dir}")
        final_signals = run_guard_stage(
            raw_signals, guard_dir, args.clickhouse_dsn, run_dir=run_dir,
            guard_debug_output=args.guard_debug_output,
            blocked_signals_output=args.blocked_signals_output,
            accepted_signals_output=args.accepted_signals_output
        )
    elif run_dir and (run_dir / "regime_guard_model.cbm").exists():
        log("INFO", "EXPORT", f"stage B: applying regime guard from {run_dir}")
        final_signals = run_guard_stage(
            raw_signals, run_dir, args.clickhouse_dsn, run_dir=run_dir,
            guard_debug_output=args.guard_debug_output,
            blocked_signals_output=args.blocked_signals_output,
            accepted_signals_output=args.accepted_signals_output
        )
    else:
        log("INFO", "EXPORT", "stage B: no guard model, passing all raw signals through")
        final_signals = raw_signals

    with open(args.output, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['symbol', 'timestamp'])
        for _, row in final_signals.iterrows():
            timestamp_str = pd.to_datetime(row['open_time']).strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([row['symbol'], timestamp_str])

    log("INFO", "EXPORT",
        f"done symbols={total_processed} candidates={total_candidates} "
        f"raw_signals={total_signals} final_signals={len(final_signals)} output={args.output}")


if __name__ == "__main__":
    main()
