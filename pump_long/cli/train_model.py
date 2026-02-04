import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import clickhouse_connect
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_long.features.feature_builder import PumpLongFeatureBuilder
from pump_long.infra.clickhouse import DataLoader
from pump_long.infra.logging import log
from pump_long.threshold import threshold_sweep_long, _prepare_event_data
from pump_long.evaluate import (
    evaluate_long, extract_signals_event_windows_long,
    compute_stream_metrics_long
)
from pump_long.tuning import predict_proba_long


def load_labels(path: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'event_open_time' not in df.columns:
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'event_open_time'})

    df['event_open_time'] = pd.to_datetime(df['event_open_time'], utc=True).dt.tz_localize(None)

    if start_date:
        df = df[df['event_open_time'] >= start_date]
    if end_date:
        df = df[df['event_open_time'] < end_date]

    return df


def time_split(
        points_df: pd.DataFrame,
        train_end: datetime,
        val_end: datetime
) -> pd.DataFrame:
    event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')

    conditions = [
        event_times['open_time'] < train_end,
        event_times['open_time'] < val_end
    ]
    choices = ['train', 'val']
    event_times['split'] = np.select(conditions, choices, default='test')

    event_to_split = event_times.set_index('event_id')['split']

    points_df = points_df.copy()
    points_df['split'] = points_df['event_id'].map(event_to_split)

    return points_df


def apply_embargo(
        points_df: pd.DataFrame,
        train_end: datetime,
        val_end: datetime,
        embargo_bars: int
) -> pd.DataFrame:
    embargo_delta = timedelta(minutes=embargo_bars * 15)

    event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')

    train_embargo_start = train_end - embargo_delta
    train_embargo_end = train_end + embargo_delta
    val_embargo_start = val_end - embargo_delta
    val_embargo_end = val_end + embargo_delta

    in_train_embargo = (event_times['open_time'] >= train_embargo_start) & (
            event_times['open_time'] < train_embargo_end)
    in_val_embargo = (event_times['open_time'] >= val_embargo_start) & (event_times['open_time'] < val_embargo_end)

    events_to_remove = event_times[in_train_embargo | in_val_embargo]['event_id']

    points_df = points_df[~points_df['event_id'].isin(events_to_remove)].copy()

    return points_df


def clip_points_to_split_bounds(
        points_df: pd.DataFrame,
        train_end: datetime,
        val_end: datetime,
        test_end: datetime = None
) -> pd.DataFrame:
    points_df = points_df.copy()

    train_mask = points_df['split'] == 'train'
    points_df = points_df[~(train_mask & (points_df['open_time'] >= train_end))]

    val_mask = points_df['split'] == 'val'
    points_df = points_df[~(val_mask & (points_df['open_time'] < train_end))]
    points_df = points_df[~(val_mask & (points_df['open_time'] >= val_end))]

    test_mask = points_df['split'] == 'test'
    points_df = points_df[~(test_mask & (points_df['open_time'] < val_end))]
    if test_end:
        points_df = points_df[~(test_mask & (points_df['open_time'] >= test_end))]

    return points_df.reset_index(drop=True)


def get_split_info(points_df: pd.DataFrame) -> dict:
    info = {}

    for split_name in ['train', 'val', 'test']:
        split_points = points_df[points_df['split'] == split_name]
        n_events = split_points['event_id'].nunique()
        n_points = len(split_points)
        n_positive = len(split_points[split_points['y'] == 1])
        n_negative = len(split_points[split_points['y'] == 0])

        info[split_name] = {
            'n_events': n_events,
            'n_points': n_points,
            'n_positive': n_positive,
            'n_negative': n_negative
        }

    return info


def train_catboost_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42,
        pos_weight: float = None
) -> CatBoostClassifier:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

    sample_weights = None
    if pos_weight is not None and pos_weight > 0:
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    train_pool = Pool(X_train, y_train, weight=sample_weights)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        early_stopping_rounds=early_stopping_rounds,
        thread_count=thread_count,
        random_seed=seed,
        verbose=100,
        eval_metric='Logloss',
        use_best_model=True
    )

    model.fit(train_pool, eval_set=val_pool)

    return model


def get_feature_importance(model: CatBoostClassifier, feature_columns: list) -> pd.DataFrame:
    importance = model.get_feature_importance()

    df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def calibrate_threshold(
        model,
        features_df: pd.DataFrame,
        feature_columns: list,
        signal_rule: str,
        alpha_hitM1: float,
        beta_early: float,
        beta_late: float,
        gamma_miss: float,
        grid_from: float = 0.01,
        grid_to: float = 0.99,
        grid_step: float = 0.01
) -> tuple:
    val_df = features_df[features_df['split'] == 'val'].copy()

    if len(val_df) == 0:
        return 0.1, pd.DataFrame()

    val_predictions = predict_proba_long(model, val_df, feature_columns)
    val_event_data = _prepare_event_data(val_predictions)

    best_threshold, sweep_df = threshold_sweep_long(
        val_predictions,
        grid_from=grid_from,
        grid_to=grid_to,
        grid_step=grid_step,
        alpha_hitM1=alpha_hitM1,
        beta_early=beta_early,
        beta_late=beta_late,
        gamma_miss=gamma_miss,
        signal_rule=signal_rule,
        event_data=val_event_data
    )

    return best_threshold, sweep_df


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def _get_all_symbols(ch_dsn: str, start_dt: datetime, end_dt: datetime) -> list:
    parsed = urlparse(ch_dsn)
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

    query = """
    SELECT DISTINCT symbol
    FROM bybit.candles
    WHERE open_time >= %(start)s
      AND open_time < %(end)s
      AND interval = 1
      AND symbol LIKE '%%USDT'
    ORDER BY symbol
    """
    result = client.query(query, parameters={
        "start": start_dt,
        "end": end_dt
    })
    return [row[0] for row in result.result_rows]


def _scan_symbol_prodlike_stream(args: tuple) -> pd.DataFrame:
    ch_dsn, symbol, start_dt, end_dt, model_path, window_bars, warmup_bars, feature_set, feature_columns = args

    model = CatBoostClassifier()
    model.load_model(model_path)

    builder = PumpLongFeatureBuilder(
        ch_dsn=ch_dsn,
        window_bars=window_bars,
        warmup_bars=warmup_bars,
        feature_set=feature_set
    )

    loader = DataLoader(ch_dsn)
    buffer_bars = warmup_bars + window_bars + 100
    load_start = start_dt - timedelta(minutes=buffer_bars * 15)

    df = loader.load_candles_range(symbol, load_start, end_dt)

    if df.empty or len(df) < warmup_bars + window_bars:
        return pd.DataFrame(columns=['symbol', 'open_time', 'p_long', 'dollar_vol_prev', 'vol_ratio_prev'])

    scan_times = []
    for ts in df.index:
        if ts >= start_dt and ts < end_dt:
            scan_times.append(ts + timedelta(minutes=15))

    if not scan_times:
        return pd.DataFrame(columns=['symbol', 'open_time', 'p_long', 'dollar_vol_prev', 'vol_ratio_prev'])

    features_list = builder.build_many_for_inference(df, symbol, scan_times)

    if not features_list:
        return pd.DataFrame(columns=['symbol', 'open_time', 'p_long', 'dollar_vol_prev', 'vol_ratio_prev'])

    features_df = pd.DataFrame(features_list)
    features_df['symbol'] = symbol

    available_features = [c for c in feature_columns if c in features_df.columns]
    if len(available_features) < len(feature_columns):
        return pd.DataFrame(columns=['symbol', 'open_time', 'p_long', 'dollar_vol_prev', 'vol_ratio_prev'])

    X = features_df[feature_columns]
    p_long = model.predict_proba(X)[:, 1]

    df_shifted = df.shift(1)
    dollar_vol_prev_map = (df_shifted['close'] * df_shifted['volume']).to_dict()

    vol_median = df['volume'].rolling(window=50).median().shift(1)
    vol_ratio_prev_map = (df['volume'].shift(1) / vol_median).to_dict()

    open_times = features_df['open_time'].values
    dollar_vol_prev = []
    vol_ratio_prev = []

    for ot in open_times:
        bucket_start = pd.Timestamp(ot) - timedelta(minutes=15)
        dollar_vol_prev.append(dollar_vol_prev_map.get(bucket_start, np.nan))
        vol_ratio_prev.append(vol_ratio_prev_map.get(bucket_start, np.nan))

    stream_df = pd.DataFrame({
        'symbol': symbol,
        'open_time': open_times,
        'p_long': p_long,
        'dollar_vol_prev': dollar_vol_prev,
        'vol_ratio_prev': vol_ratio_prev
    })

    return stream_df.sort_values('open_time').reset_index(drop=True)


def extract_signals_from_stream(
        stream_df: pd.DataFrame,
        threshold: float,
        cooldown_bars: int,
        gate_min_dollar_vol: float = None,
        gate_min_vol_ratio: float = None
) -> pd.DataFrame:
    signals = []

    for symbol, group in stream_df.groupby('symbol'):
        group = group.sort_values('open_time').reset_index(drop=True)
        p_arr = group['p_long'].values
        times_arr = group['open_time'].values
        dollar_vol_arr = group['dollar_vol_prev'].values if 'dollar_vol_prev' in group.columns else None
        vol_ratio_arr = group['vol_ratio_prev'].values if 'vol_ratio_prev' in group.columns else None

        n = len(group)
        last_signal_idx = -cooldown_bars - 1

        for i in range(1, n):
            prev_p = p_arr[i - 1]
            curr_p = p_arr[i]

            if np.isnan(prev_p) or np.isnan(curr_p):
                continue

            if gate_min_dollar_vol is not None and dollar_vol_arr is not None:
                if np.isnan(dollar_vol_arr[i]) or dollar_vol_arr[i] < gate_min_dollar_vol:
                    continue

            if gate_min_vol_ratio is not None and vol_ratio_arr is not None:
                if np.isnan(vol_ratio_arr[i]) or vol_ratio_arr[i] < gate_min_vol_ratio:
                    continue

            is_cross_up = prev_p < threshold <= curr_p
            cooldown_ok = (i - last_signal_idx) > cooldown_bars

            if is_cross_up and cooldown_ok:
                signals.append({
                    'open_time': times_arr[i],
                    'symbol': symbol,
                    'p_long': float(curr_p),
                    'prev_p_long': float(prev_p),
                    'threshold': threshold,
                    'rule': 'cross_up',
                    'cooldown_bars': cooldown_bars
                })
                last_signal_idx = i

    if not signals:
        return pd.DataFrame(columns=['open_time', 'symbol', 'p_long', 'prev_p_long', 'threshold', 'rule', 'cooldown_bars'])

    return pd.DataFrame(signals).sort_values(['open_time', 'symbol']).reset_index(drop=True)


def run_stream_scan(
        ch_dsn: str,
        model_path: str,
        scan_start: datetime,
        scan_end: datetime,
        window_bars: int,
        warmup_bars: int,
        feature_set: str,
        feature_columns: list,
        scan_workers: int
) -> pd.DataFrame:
    log("INFO", "SCAN", f"stream scan period: [{scan_start}, {scan_end})")

    symbols = _get_all_symbols(ch_dsn, scan_start, scan_end)
    log("INFO", "SCAN", f"found {len(symbols)} USDT symbols")

    tasks = [
        (
            ch_dsn,
            symbol,
            scan_start,
            scan_end,
            str(model_path),
            window_bars,
            warmup_bars,
            feature_set,
            feature_columns
        )
        for symbol in symbols
    ]

    all_streams = []
    processed = 0

    with ProcessPoolExecutor(max_workers=scan_workers) as executor:
        for stream_df in executor.map(_scan_symbol_prodlike_stream, tasks):
            if not stream_df.empty:
                all_streams.append(stream_df)
            processed += 1
            if processed % 50 == 0:
                log("INFO", "SCAN", f"processed {processed}/{len(symbols)} symbols")

    if all_streams:
        combined_stream = pd.concat(all_streams, ignore_index=True)
        combined_stream = combined_stream.sort_values(['open_time', 'symbol']).reset_index(drop=True)
    else:
        combined_stream = pd.DataFrame(columns=['symbol', 'open_time', 'p_long', 'dollar_vol_prev', 'vol_ratio_prev'])

    log("INFO", "SCAN", f"total stream rows: {len(combined_stream)}")
    return combined_stream


def stream_threshold_sweep(
        stream_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        scan_start: datetime,
        scan_end: datetime,
        cooldown_bars: int,
        match_before_bars: int,
        match_after_bars: int,
        gate_min_dollar_vol: float = None,
        gate_min_vol_ratio: float = None,
        target_signals_per_day_min: float = 1.0,
        target_signals_per_day_max: float = 5.0,
        thr_objective: str = "precision",
        grid_from: float = 0.01,
        grid_to: float = 0.99,
        grid_step: float = 0.01
) -> tuple:
    holdout_seconds = (scan_end - scan_start).total_seconds()
    holdout_days = holdout_seconds / 86400
    if holdout_days <= 0:
        holdout_days = 1

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    sweep_results = []

    for thr in thresholds:
        signals_df = extract_signals_from_stream(
            stream_df,
            threshold=thr,
            cooldown_bars=cooldown_bars,
            gate_min_dollar_vol=gate_min_dollar_vol,
            gate_min_vol_ratio=gate_min_vol_ratio
        )

        metrics = compute_stream_metrics_long(
            signals_df,
            labels_df,
            scan_start,
            scan_end,
            window_before=match_before_bars,
            window_after=match_after_bars
        )

        sweep_results.append({
            'threshold': round(thr, 3),
            'signals_per_day': metrics['signals_per_day'],
            'precision': metrics['precision'],
            'event_recall': metrics['event_recall'],
            'tp_signals': metrics['tp_signals'],
            'fp_signals': metrics['fp_signals'],
            'total_signals': metrics['total_signals'],
            'caught_events': metrics['caught_events'],
            'total_events': metrics['total_events']
        })

    sweep_df = pd.DataFrame(sweep_results)

    valid_rows = sweep_df[
        (sweep_df['signals_per_day'] >= target_signals_per_day_min) &
        (sweep_df['signals_per_day'] <= target_signals_per_day_max)
    ]

    if valid_rows.empty:
        closest_idx = (sweep_df['signals_per_day'] - target_signals_per_day_max).abs().idxmin()
        best_threshold = sweep_df.loc[closest_idx, 'threshold']
        log("WARN", "SWEEP", f"no threshold in target range, using closest: {best_threshold}")
    else:
        if thr_objective == "precision":
            best_idx = valid_rows['precision'].idxmax()
        else:
            best_idx = valid_rows['precision'].idxmax()
        best_threshold = sweep_df.loc[best_idx, 'threshold']

    return best_threshold, sweep_df


def run_holdout_scan(
        ch_dsn: str,
        model_path: str,
        threshold: float,
        holdout_start: datetime,
        holdout_end: datetime,
        cooldown_bars: int,
        window_bars: int,
        warmup_bars: int,
        feature_set: str,
        feature_columns: list,
        scan_workers: int,
        labels_path: str = None,
        match_before_bars: int = 4,
        match_after_bars: int = 0,
        gate_min_dollar_vol: float = None,
        gate_min_vol_ratio: float = None
) -> tuple:
    stream_df = run_stream_scan(
        ch_dsn=ch_dsn,
        model_path=model_path,
        scan_start=holdout_start,
        scan_end=holdout_end,
        window_bars=window_bars,
        warmup_bars=warmup_bars,
        feature_set=feature_set,
        feature_columns=feature_columns,
        scan_workers=scan_workers
    )

    signals_df = extract_signals_from_stream(
        stream_df,
        threshold=threshold,
        cooldown_bars=cooldown_bars,
        gate_min_dollar_vol=gate_min_dollar_vol,
        gate_min_vol_ratio=gate_min_vol_ratio
    )

    log("INFO", "SCAN", f"total signals: {len(signals_df)}")

    stream_metrics = None
    if labels_path:
        log("INFO", "SCAN", "computing stream metrics with label matching")
        labels_df = load_labels(labels_path, holdout_start, holdout_end)
        labels_df = labels_df[labels_df['pump_la_type'] == 'A']
        log("INFO", "SCAN", f"loaded {len(labels_df)} type-A events in holdout period")

        stream_metrics = compute_stream_metrics_long(
            signals_df,
            labels_df,
            holdout_start,
            holdout_end,
            window_before=match_before_bars,
            window_after=match_after_bars
        )

        log("INFO", "SCAN", f"stream metrics:")
        log("INFO", "SCAN", f"  signals/day: {stream_metrics['signals_per_day']}")
        log("INFO", "SCAN", f"  precision: {stream_metrics['precision']:.4f}")
        log("INFO", "SCAN", f"  event_recall: {stream_metrics['event_recall']:.4f}")
        log("INFO", "SCAN", f"  TP: {stream_metrics['tp_signals']}, FP: {stream_metrics['fp_signals']}")
        log("INFO", "SCAN", f"  caught/total events: {stream_metrics['caught_events']}/{stream_metrics['total_events']}")

    return signals_df, stream_metrics


def main():
    parser = argparse.ArgumentParser(description="Train pump long model and optionally run holdout scan")

    parser.add_argument("--dataset-parquet", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)

    parser.add_argument("--train-end", type=str, required=True)
    parser.add_argument("--val-end", type=str, required=True)
    parser.add_argument("--embargo-bars", type=int, default=80)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--signal-rule", type=str, choices=["first_cross"], default="first_cross")
    parser.add_argument("--alpha-hitM1", type=float, default=0.8)
    parser.add_argument("--beta-early", type=float, default=1.0)
    parser.add_argument("--beta-late", type=float, default=3.0)
    parser.add_argument("--gamma-miss", type=float, default=1.0)
    parser.add_argument("--thr-grid-from", type=float, default=0.01)
    parser.add_argument("--thr-grid-to", type=float, default=0.99)
    parser.add_argument("--thr-grid-step", type=float, default=0.01)

    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--holdout-start", type=str, default=None)
    parser.add_argument("--holdout-end", type=str, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=8)
    parser.add_argument("--scan-workers", type=int, default=4)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--neg-before", type=int, default=60)
    parser.add_argument("--neg-after", type=int, default=10)
    parser.add_argument("--match-before-bars", type=int, default=4)
    parser.add_argument("--match-after-bars", type=int, default=0)

    parser.add_argument("--pos-weight", type=float, default=None)

    parser.add_argument("--target-signals-per-day-min", type=float, default=1.0)
    parser.add_argument("--target-signals-per-day-max", type=float, default=5.0)
    parser.add_argument("--thr-objective", type=str, choices=["precision"], default="precision")

    parser.add_argument("--gate-min-dollar-vol", type=float, default=None)
    parser.add_argument("--gate-min-vol-ratio", type=float, default=None)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.run_name:
        run_dir = Path(args.out_dir) / args.run_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(args.out_dir) / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", "TRAIN", f"run_dir={run_dir}")

    log("INFO", "TRAIN", f"loading manifest from {args.manifest}")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    feature_columns = manifest['feature_columns']
    log("INFO", "TRAIN", f"feature_columns from manifest: {len(feature_columns)} features")

    log("INFO", "TRAIN", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)
    log("INFO", "TRAIN", f"loaded {len(features_df)} rows")

    missing_cols = [c for c in feature_columns if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols[:10]}... ({len(missing_cols)} total)")

    train_end = parse_date_exclusive(args.train_end)
    val_end = parse_date_exclusive(args.val_end)

    features_df = time_split(features_df, train_end, val_end)

    if args.embargo_bars > 0:
        features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

    features_df = clip_points_to_split_bounds(features_df, train_end, val_end)

    split_info = get_split_info(features_df)
    splits_path = run_dir / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(split_info, f, indent=2, default=str)
    log("INFO", "TRAIN",
        f"split info: train={split_info['train']['n_events']} val={split_info['val']['n_events']} test={split_info['test']['n_events']} events")

    log("INFO", "TRAIN", "training CatBoost model")
    model = train_catboost_model(
        features_df,
        feature_columns,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        thread_count=args.thread_count,
        seed=args.seed,
        pos_weight=args.pos_weight
    )

    model_path = run_dir / "model.cbm"
    model.save_model(str(model_path))
    log("INFO", "TRAIN", f"model saved to {model_path}")

    feature_cols_path = run_dir / "feature_columns.json"
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    log("INFO", "TRAIN", f"feature_columns saved to {feature_cols_path}")

    importance_df = get_feature_importance(model, feature_columns)
    importance_path = run_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    log("INFO", "TRAIN", "calibrating threshold on val set (event-window)")
    best_threshold_event, sweep_df_event = calibrate_threshold(
        model,
        features_df,
        feature_columns,
        args.signal_rule,
        args.alpha_hitM1,
        args.beta_early,
        args.beta_late,
        args.gamma_miss,
        args.thr_grid_from,
        args.thr_grid_to,
        args.thr_grid_step
    )

    if not sweep_df_event.empty:
        sweep_path = run_dir / "threshold_sweep_event.csv"
        sweep_df_event.to_csv(sweep_path, index=False)
    log("INFO", "TRAIN", f"event-window threshold: {best_threshold_event:.3f}")

    gates_config = {
        'gate_min_dollar_vol': args.gate_min_dollar_vol,
        'gate_min_vol_ratio': args.gate_min_vol_ratio
    }
    gates_path = run_dir / "gates.json"
    with open(gates_path, 'w') as f:
        json.dump(gates_config, f, indent=2)

    best_threshold = best_threshold_event
    stream_calibrated = False

    if args.clickhouse_dsn and args.labels:
        log("INFO", "TRAIN", "running stream calibration on VAL period")

        val_scan_start = train_end
        val_scan_end = val_end

        val_stream_df = run_stream_scan(
            ch_dsn=args.clickhouse_dsn,
            model_path=str(model_path),
            scan_start=val_scan_start,
            scan_end=val_scan_end,
            window_bars=manifest.get('window_bars', 60),
            warmup_bars=manifest.get('warmup_bars', 150),
            feature_set=manifest.get('feature_set', 'extended'),
            feature_columns=feature_columns,
            scan_workers=args.scan_workers
        )

        val_labels_df = load_labels(args.labels, val_scan_start, val_scan_end)
        val_labels_df = val_labels_df[val_labels_df['pump_la_type'] == 'A']
        log("INFO", "TRAIN", f"loaded {len(val_labels_df)} type-A events for VAL stream calibration")

        if len(val_stream_df) > 0 and len(val_labels_df) > 0:
            best_threshold_stream, sweep_df_stream = stream_threshold_sweep(
                stream_df=val_stream_df,
                labels_df=val_labels_df,
                scan_start=val_scan_start,
                scan_end=val_scan_end,
                cooldown_bars=args.cooldown_bars,
                match_before_bars=args.match_before_bars,
                match_after_bars=args.match_after_bars,
                gate_min_dollar_vol=args.gate_min_dollar_vol,
                gate_min_vol_ratio=args.gate_min_vol_ratio,
                target_signals_per_day_min=args.target_signals_per_day_min,
                target_signals_per_day_max=args.target_signals_per_day_max,
                thr_objective=args.thr_objective,
                grid_from=args.thr_grid_from,
                grid_to=args.thr_grid_to,
                grid_step=args.thr_grid_step
            )

            sweep_stream_path = run_dir / "threshold_sweep_stream.csv"
            sweep_df_stream.to_csv(sweep_stream_path, index=False)

            best_threshold = best_threshold_stream
            stream_calibrated = True
            log("INFO", "TRAIN", f"stream-calibrated threshold: {best_threshold:.3f}")

            best_threshold_stream_data = {
                'threshold': best_threshold_stream,
                'signal_rule': 'cross_up',
                'cooldown_bars': args.cooldown_bars,
                'match_before_bars': args.match_before_bars,
                'match_after_bars': args.match_after_bars,
                'gate_min_dollar_vol': args.gate_min_dollar_vol,
                'gate_min_vol_ratio': args.gate_min_vol_ratio,
                'target_signals_per_day_min': args.target_signals_per_day_min,
                'target_signals_per_day_max': args.target_signals_per_day_max,
                'calibration_period': f"{val_scan_start.date()} to {val_scan_end.date()}"
            }
            threshold_stream_path = run_dir / "best_threshold_stream.json"
            with open(threshold_stream_path, 'w') as f:
                json.dump(best_threshold_stream_data, f, indent=2, default=str)

            val_signals_stream = extract_signals_from_stream(
                val_stream_df,
                threshold=best_threshold,
                cooldown_bars=args.cooldown_bars,
                gate_min_dollar_vol=args.gate_min_dollar_vol,
                gate_min_vol_ratio=args.gate_min_vol_ratio
            )

            val_signals_stream_path = run_dir / "predicted_signals_val_stream.csv"
            val_signals_stream.to_csv(val_signals_stream_path, index=False)

            val_stream_metrics = compute_stream_metrics_long(
                val_signals_stream,
                val_labels_df,
                val_scan_start,
                val_scan_end,
                window_before=args.match_before_bars,
                window_after=args.match_after_bars
            )

            val_stream_metrics_path = run_dir / "stream_metrics_val.json"
            with open(val_stream_metrics_path, 'w') as f:
                json.dump(val_stream_metrics, f, indent=2, default=str)

            log("INFO", "TRAIN", f"VAL stream: signals/day={val_stream_metrics['signals_per_day']}, precision={val_stream_metrics['precision']:.4f}, recall={val_stream_metrics['event_recall']:.4f}")

    threshold_path = run_dir / "best_threshold.json"
    threshold_data = {
        'threshold': best_threshold,
        'signal_rule': args.signal_rule if not stream_calibrated else 'cross_up',
        'stream_calibrated': stream_calibrated,
        'cooldown_bars': args.cooldown_bars,
        'match_before_bars': args.match_before_bars,
        'match_after_bars': args.match_after_bars,
        'gate_min_dollar_vol': args.gate_min_dollar_vol,
        'gate_min_vol_ratio': args.gate_min_vol_ratio
    }
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2, default=str)
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    log("INFO", "TRAIN", "evaluating on val set (event-window)")
    val_predictions = predict_proba_long(
        model,
        features_df[features_df['split'] == 'val'],
        feature_columns
    )
    val_event_data = _prepare_event_data(val_predictions)
    val_metrics = evaluate_long(val_predictions, best_threshold, signal_rule=args.signal_rule,
                                event_data=val_event_data)

    val_metrics_path = run_dir / "metrics_val.json"
    with open(val_metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2, default=str)
    log("INFO", "TRAIN",
        f"val metrics: hit0={val_metrics['event_level']['hit0_rate']:.3f} hitM1={val_metrics['event_level']['hitM1_rate']:.3f} late={val_metrics['event_level']['late_rate']:.3f}")

    log("INFO", "TRAIN", "evaluating on test set (event-window)")
    test_predictions = predict_proba_long(
        model,
        features_df[features_df['split'] == 'test'],
        feature_columns
    )
    test_event_data = _prepare_event_data(test_predictions)
    test_metrics = evaluate_long(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                 event_data=test_event_data)

    test_metrics_path = run_dir / "metrics_test.json"
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} hitM1={test_metrics['event_level']['hitM1_rate']:.3f} late={test_metrics['event_level']['late_rate']:.3f}")

    signals_df = extract_signals_event_windows_long(test_predictions, best_threshold, signal_rule=args.signal_rule)
    signals_path = run_dir / "predicted_signals_event_windows.csv"
    signals_df.to_csv(signals_path, index=False)
    log("INFO", "TRAIN", f"saved {len(signals_df)} event-window signals")

    if args.clickhouse_dsn and args.labels:
        test_scan_start = val_end
        test_times = features_df[features_df['split'] == 'test']['open_time']
        if len(test_times) > 0:
            test_scan_end = test_times.max() + timedelta(days=1)
        else:
            test_scan_end = val_end + timedelta(days=30)

        log("INFO", "TRAIN", "running prod-like scan on TEST period")

        test_stream_df = run_stream_scan(
            ch_dsn=args.clickhouse_dsn,
            model_path=str(model_path),
            scan_start=test_scan_start,
            scan_end=test_scan_end,
            window_bars=manifest.get('window_bars', 60),
            warmup_bars=manifest.get('warmup_bars', 150),
            feature_set=manifest.get('feature_set', 'extended'),
            feature_columns=feature_columns,
            scan_workers=args.scan_workers
        )

        test_signals_stream = extract_signals_from_stream(
            test_stream_df,
            threshold=best_threshold,
            cooldown_bars=args.cooldown_bars,
            gate_min_dollar_vol=args.gate_min_dollar_vol,
            gate_min_vol_ratio=args.gate_min_vol_ratio
        )

        test_signals_stream_path = run_dir / "predicted_signals_test_stream.csv"
        test_signals_stream.to_csv(test_signals_stream_path, index=False)
        log("INFO", "TRAIN", f"saved {len(test_signals_stream)} test stream signals")

        test_labels_df = load_labels(args.labels, test_scan_start, test_scan_end)
        test_labels_df = test_labels_df[test_labels_df['pump_la_type'] == 'A']

        if len(test_labels_df) > 0:
            test_stream_metrics = compute_stream_metrics_long(
                test_signals_stream,
                test_labels_df,
                test_scan_start,
                test_scan_end,
                window_before=args.match_before_bars,
                window_after=args.match_after_bars
            )

            test_stream_metrics_path = run_dir / "stream_metrics_test.json"
            with open(test_stream_metrics_path, 'w') as f:
                json.dump(test_stream_metrics, f, indent=2, default=str)

            log("INFO", "TRAIN", f"TEST stream: signals/day={test_stream_metrics['signals_per_day']}, precision={test_stream_metrics['precision']:.4f}, recall={test_stream_metrics['event_recall']:.4f}")

    if args.holdout_start and args.holdout_end and args.clickhouse_dsn:
        holdout_start = datetime.strptime(args.holdout_start, '%Y-%m-%d')
        holdout_end = parse_date_exclusive(args.holdout_end)

        holdout_signals_df, stream_metrics = run_holdout_scan(
            ch_dsn=args.clickhouse_dsn,
            model_path=str(model_path),
            threshold=best_threshold,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            cooldown_bars=args.cooldown_bars,
            window_bars=manifest.get('window_bars', 60),
            warmup_bars=manifest.get('warmup_bars', 150),
            feature_set=manifest.get('feature_set', 'extended'),
            feature_columns=feature_columns,
            scan_workers=args.scan_workers,
            labels_path=args.labels,
            match_before_bars=args.match_before_bars,
            match_after_bars=args.match_after_bars,
            gate_min_dollar_vol=args.gate_min_dollar_vol,
            gate_min_vol_ratio=args.gate_min_vol_ratio
        )

        holdout_signals_path = run_dir / "predicted_signals_holdout.csv"
        holdout_signals_df.to_csv(holdout_signals_path, index=False)
        log("INFO", "TRAIN", f"saved {len(holdout_signals_df)} holdout signals")

        if stream_metrics:
            stream_metrics_path = run_dir / "stream_metrics_holdout.json"
            with open(stream_metrics_path, 'w') as f:
                json.dump(stream_metrics, f, indent=2, default=str)

    config = vars(args)
    config['run_dir'] = str(run_dir)
    config['created_at'] = datetime.now().isoformat()
    config_path = run_dir / "run_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    log("INFO", "TRAIN", f"done. artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
