import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd
from catboost import CatBoostClassifier

from pump_end.features.feature_builder import PumpFeatureBuilder
from pump_end.ml.train import get_feature_columns
from pump_end.ml.feature_schema import prune_feature_columns


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


def get_buckets_for_symbol(client, symbol: str, start_dt: datetime, end_dt: datetime) -> list:
    query = """
            SELECT DISTINCT toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket
            FROM bybit.candles
            WHERE symbol = %(symbol)s
              AND open_time >= %(start)s
              AND open_time < %(end)s
              AND interval = 1
            ORDER BY bucket \
            """
    result = client.query(query, parameters={
        "symbol": symbol,
        "start": start_dt,
        "end": end_dt
    })
    return [row[0] for row in result.result_rows]


def apply_signal_rule_first_cross(
        symbol_df: pd.DataFrame,
        threshold: float
) -> list:
    signals = []
    below_threshold = True

    for _, row in symbol_df.iterrows():
        if row['p_end'] >= threshold:
            if below_threshold:
                signals.append({
                    'symbol': row['symbol'],
                    'timestamp': row['open_time']
                })
                below_threshold = False
        else:
            below_threshold = True

    return signals


def apply_signal_rule_pending_turn_down(
        symbol_df: pd.DataFrame,
        threshold: float,
        min_pending_bars: int,
        drop_delta: float
) -> list:
    signals = []
    pending_count = 0
    prev_p_end = None

    for _, row in symbol_df.iterrows():
        p_end = row['p_end']

        if p_end >= threshold:
            pending_count += 1
            if pending_count >= min_pending_bars and prev_p_end is not None:
                drop = prev_p_end - p_end
                if p_end < prev_p_end and drop >= drop_delta:
                    signals.append({
                        'symbol': row['symbol'],
                        'timestamp': row['open_time']
                    })
                    pending_count = 0
        else:
            pending_count = 0

        prev_p_end = p_end

    return signals


def apply_signal_rule_argmax_per_event(
        symbol_df: pd.DataFrame,
        threshold: float
) -> list:
    signals = []
    segment_start = None
    segment_max_p = -1
    segment_max_row = None

    for idx, row in symbol_df.iterrows():
        p_end = row['p_end']

        if p_end >= threshold:
            if segment_start is None:
                segment_start = idx
                segment_max_p = p_end
                segment_max_row = row
            else:
                if p_end > segment_max_p:
                    segment_max_p = p_end
                    segment_max_row = row
        else:
            if segment_start is not None and segment_max_row is not None:
                signals.append({
                    'symbol': segment_max_row['symbol'],
                    'timestamp': segment_max_row['open_time']
                })
            segment_start = None
            segment_max_p = -1
            segment_max_row = None

    if segment_start is not None and segment_max_row is not None:
        signals.append({
            'symbol': segment_max_row['symbol'],
            'timestamp': segment_max_row['open_time']
        })

    return signals


def apply_signal_rule(
        pred_df: pd.DataFrame,
        signal_rule: str,
        threshold: float,
        min_pending_bars: int,
        drop_delta: float
) -> list:
    all_signals = []

    for symbol, group in pred_df.groupby('symbol'):
        symbol_df = group.sort_values('open_time').reset_index(drop=True)

        if signal_rule == 'first_cross':
            signals = apply_signal_rule_first_cross(symbol_df, threshold)
        elif signal_rule == 'pending_turn_down':
            signals = apply_signal_rule_pending_turn_down(
                symbol_df, threshold, min_pending_bars, drop_delta
            )
        elif signal_rule == 'argmax_per_event':
            signals = apply_signal_rule_argmax_per_event(symbol_df, threshold)
        else:
            signals = apply_signal_rule_pending_turn_down(
                symbol_df, threshold, min_pending_bars, drop_delta
            )

        all_signals.extend(signals)

    return all_signals


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

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M:%S')
    model_dir = Path(args.model_dir)

    log("INFO", "EXPORT", f"loading artifacts from {model_dir}")
    run_config = load_run_config(model_dir)
    threshold_config = load_threshold_config(model_dir)
    model = load_model(model_dir)

    window_bars = run_config.get('window_bars', 30)
    warmup_bars = run_config.get('warmup_bars', 150)
    feature_set = run_config.get('feature_set', 'base')
    do_prune = run_config.get('prune_features', False)

    threshold = threshold_config['threshold']
    signal_rule = threshold_config.get('signal_rule', 'pending_turn_down')
    min_pending_bars = threshold_config.get('min_pending_bars', 1)
    drop_delta = threshold_config.get('drop_delta', 0.0)

    log("INFO", "EXPORT",
        f"config: window_bars={window_bars} warmup_bars={warmup_bars} feature_set={feature_set} prune={do_prune}")
    log("INFO", "EXPORT",
        f"threshold={threshold} signal_rule={signal_rule} min_pending_bars={min_pending_bars} drop_delta={drop_delta}")

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

    all_feature_inputs = []

    for symbol in symbols:
        buckets = get_buckets_for_symbol(client, symbol, start_dt, end_dt)
        for bucket in buckets:
            all_feature_inputs.append({
                'symbol': symbol,
                'event_open_time': bucket,
                'pump_la_type': 'A',
                'runup_pct': 0
            })

    if not all_feature_inputs:
        log("WARN", "EXPORT", "no buckets found")
        return

    log("INFO", "EXPORT", f"total buckets to process: {len(all_feature_inputs)}")

    feature_input_df = pd.DataFrame(all_feature_inputs)

    log("INFO", "EXPORT", "building features")
    builder = PumpFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=window_bars,
        warmup_bars=warmup_bars,
        feature_set=feature_set
    )

    features_df = builder.build(feature_input_df, max_workers=4)

    if features_df.empty:
        log("WARN", "EXPORT", "no features extracted")
        return

    log("INFO", "EXPORT", f"features shape: {features_df.shape}")

    feature_columns = get_feature_columns(features_df)

    if do_prune:
        original_count = len(feature_columns)
        feature_columns = prune_feature_columns(feature_columns)
        log("INFO", "EXPORT", f"pruned features: {original_count} -> {len(feature_columns)}")

    X = features_df[feature_columns]
    p_end = model.predict_proba(X)[:, 1]

    pred_df = features_df[['symbol', 'open_time']].copy()
    pred_df['p_end'] = p_end

    log("INFO", "EXPORT", "applying signal rule")
    signals = apply_signal_rule(
        pred_df,
        signal_rule,
        threshold,
        min_pending_bars,
        drop_delta
    )

    if not signals:
        log("WARN", "EXPORT", "no signals generated")
        return

    signals_df = pd.DataFrame(signals)
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    signals_df = signals_df.drop_duplicates(subset=['symbol', 'timestamp'])
    signals_df = signals_df.sort_values(['timestamp', 'symbol'])
    signals_df['timestamp'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    signals_df[['symbol', 'timestamp']].to_csv(args.output, index=False)

    log("INFO", "EXPORT", f"exported {len(signals_df)} signals to {args.output}")


if __name__ == "__main__":
    main()
