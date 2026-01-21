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

from src.shared.clickhouse import DataLoader
from src.shared.pump_end.feature_builder import PumpFeatureBuilder
from src.dev.pump_long.labeler import PumpStartLabelerLookahead
from src.dev.pump_long.threshold import threshold_sweep_long, _prepare_event_data
from src.dev.pump_long.evaluate import evaluate_long, extract_signals_long
from src.dev.pump_long.tuning import (
    tune_model_long, train_final_model_long, predict_proba_long,
    generate_walk_forward_folds, compute_sample_weights
)


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class RunArtifacts:
    def __init__(self, out_dir: str, run_name: str = None):
        if run_name:
            self.run_dir = Path(out_dir) / run_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = Path(out_dir) / f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict):
        path = self.run_dir / "run_config.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save_labels(self, df: pd.DataFrame):
        path = self.run_dir / "pump_long_labels.csv"
        df.to_csv(path, index=False)

    def save_training_points(self, df: pd.DataFrame):
        path = self.run_dir / "training_points.parquet"
        df.to_parquet(path, index=False)

    def save_features(self, df: pd.DataFrame):
        path = self.run_dir / "features.parquet"
        df.to_parquet(path, index=False)

    def save_splits(self, splits_info: dict):
        path = self.run_dir / "splits.json"
        with open(path, 'w') as f:
            json.dump(splits_info, f, indent=2, default=str)

    def save_model(self, model):
        path = self.run_dir / "catboost_model.cbm"
        model.save_model(str(path))

    def save_threshold_sweep(self, df: pd.DataFrame):
        path = self.run_dir / "threshold_sweep.csv"
        df.to_csv(path, index=False)

    def save_metrics(self, metrics: dict, split_name: str):
        path = self.run_dir / f"metrics_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_feature_importance(self, df: pd.DataFrame):
        path = self.run_dir / "feature_importance.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_holdout.csv"
        df.to_csv(path, index=False)

    def save_predictions(self, df: pd.DataFrame, split_name: str):
        path = self.run_dir / f"predictions_{split_name}.parquet"
        df.to_parquet(path, index=False)

    def save_best_params(self, params: dict):
        path = self.run_dir / "best_params.json"
        with open(path, 'w') as f:
            json.dump(params, f, indent=2, default=str)

    def save_best_threshold(self, threshold: float, signal_rule: str = 'cross_up', hysteresis_delta: float = 0.05):
        data = {'threshold': threshold, 'signal_rule': signal_rule, 'hysteresis_delta': hysteresis_delta}
        path = self.run_dir / "best_threshold.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_leaderboard(self, df: pd.DataFrame):
        path = self.run_dir / "leaderboard.csv"
        df.to_csv(path, index=False)

    def save_cv_report(self, cv_result: dict):
        path = self.run_dir / "cv_report.json"
        serializable = {
            'mean_score': cv_result.get('mean_score'),
            'std_score': cv_result.get('std_score'),
            'mean_threshold': cv_result.get('mean_threshold'),
            'fold_results': cv_result.get('fold_results', [])
        }
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

    def save_folds(self, folds: list):
        path = self.run_dir / "folds.json"
        with open(path, 'w') as f:
            json.dump(folds, f, indent=2, default=str)

    def get_path(self) -> Path:
        return self.run_dir


def _process_symbol_labels(args: tuple) -> list:
    ch_dsn, symbol, start_dt, end_dt = args

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
            SELECT toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
                   argMin(open, open_time) AS open,
        max(high) AS high,
        min(low) AS low,
        argMax(close, open_time) AS close,
        sum(volume) AS volume
            FROM bybit.candles
            WHERE symbol = %(symbol)s
              AND interval = 1
              AND open_time >= %(start)s
              AND open_time
                < %(end)s
            GROUP BY bucket
            ORDER BY bucket \
            """

    result = client.query(query, parameters={
        "symbol": symbol,
        "start": start_dt,
        "end": end_dt + timedelta(minutes=15)
    })

    if not result.result_rows:
        return []

    df = pd.DataFrame(
        result.result_rows,
        columns=["bucket", "open", "high", "low", "close", "volume"]
    )
    df["bucket"] = pd.to_datetime(df["bucket"])
    df.set_index("bucket", inplace=True)

    labeler = PumpStartLabelerLookahead()
    df = labeler.detect(df)

    labeled = df[df['pump_start_type'].notna()]
    if labeled.empty:
        return []

    labels = []
    for idx in labeled.index:
        row = labeled.loc[idx]
        labels.append({
            'symbol': symbol,
            'event_open_time': row['start_open_time'],
            'peak_open_time': row['peak_open_time'],
            'pump_la_type': row['pump_start_type'],
            'runup_pct': round(row['pump_start_runup'] * 100, 2)
        })

    return labels


def export_pump_long_labels(ch_dsn: str, start_dt: datetime, end_dt: datetime, max_workers: int = 4) -> pd.DataFrame:
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

    symbols_query = """
                    SELECT DISTINCT symbol
                    FROM bybit.candles
                    WHERE open_time >= %(start_date)s
                      AND open_time <= %(end_date)s
                      AND interval = 1
                    ORDER BY symbol \
                    """
    result = client.query(symbols_query, parameters={
        "start_date": start_dt,
        "end_date": end_dt
    })
    symbols = [row[0] for row in result.result_rows]

    if not symbols:
        return pd.DataFrame()

    log("INFO", "EXPORT", f"found {len(symbols)} symbols")

    tasks = [(ch_dsn, symbol, start_dt, end_dt) for symbol in symbols]

    all_labels = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for labels in executor.map(_process_symbol_labels, tasks):
            all_labels.extend(labels)

    if all_labels:
        df = pd.DataFrame(all_labels)
        df = df.sort_values(['symbol', 'event_open_time']).reset_index(drop=True)
        return df
    return pd.DataFrame()


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


def build_training_points(
        labels_df: pd.DataFrame,
        neg_before: int = 60,
        neg_after: int = 10,
        pos_offsets: list = None
) -> pd.DataFrame:
    if pos_offsets is None:
        pos_offsets = [0]

    events = labels_df[labels_df['pump_la_type'] == 'A'].copy()

    if events.empty:
        return pd.DataFrame()

    events['event_id'] = events['symbol'] + '|' + events['event_open_time'].dt.strftime('%Y%m%d_%H%M%S')

    all_offsets = list(range(-neg_before, 0)) + pos_offsets + list(
        range(max(pos_offsets) + 1, max(pos_offsets) + neg_after + 1))
    all_offsets = sorted(set(all_offsets))

    n_events = len(events)
    n_offsets = len(all_offsets)

    event_ids = np.repeat(events['event_id'].values, n_offsets)
    symbols = np.repeat(events['symbol'].values, n_offsets)
    base_times = np.repeat(events['event_open_time'].values, n_offsets)
    runup_pcts = np.repeat(events['runup_pct'].values if 'runup_pct' in events.columns else np.nan, n_offsets)

    offsets = np.tile(all_offsets, n_events)
    time_deltas = pd.to_timedelta(offsets * 15, unit='m')
    open_times = pd.to_datetime(base_times) + time_deltas

    pos_offsets_set = set(pos_offsets)
    y_values = np.where(np.isin(offsets, list(pos_offsets_set)), 1, 0)

    points_df = pd.DataFrame({
        'event_id': event_ids,
        'symbol': symbols,
        'open_time': open_times,
        'offset': offsets,
        'y': y_values,
        'runup_pct': runup_pcts,
        'is_background': False
    })

    return points_df


def add_background_negatives(
        points_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        num_per_symbol: int = 50,
        min_distance_bars: int = 300,
        seed: int = 42
) -> pd.DataFrame:
    if num_per_symbol <= 0:
        return points_df

    rng = np.random.default_rng(seed)

    events = labels_df[labels_df['pump_la_type'] == 'A'].copy()
    events['event_open_time'] = pd.to_datetime(events['event_open_time'])

    symbols = events['symbol'].unique()
    min_distance = timedelta(minutes=min_distance_bars * 15)

    background_points = []

    for symbol in symbols:
        symbol_events = events[events['symbol'] == symbol]['event_open_time'].values
        symbol_events = pd.to_datetime(symbol_events)

        if len(symbol_events) < 2:
            continue

        min_time = symbol_events.min()
        max_time = symbol_events.max()

        time_range = (max_time - min_time).total_seconds() / 60 / 15

        if time_range < min_distance_bars * 2:
            continue

        generated = 0
        attempts = 0
        max_attempts = num_per_symbol * 10

        while generated < num_per_symbol and attempts < max_attempts:
            attempts += 1

            random_offset = int(rng.integers(0, int(time_range)))
            candidate_time = min_time + timedelta(minutes=random_offset * 15)

            is_far = True
            for event_time in symbol_events:
                if abs((candidate_time - event_time).total_seconds()) < min_distance.total_seconds():
                    is_far = False
                    break

            if is_far:
                background_points.append({
                    'event_id': f'bg_{symbol}_{generated}',
                    'symbol': symbol,
                    'open_time': candidate_time,
                    'offset': 0,
                    'y': 0,
                    'runup_pct': 0,
                    'is_background': True
                })
                generated += 1

    if background_points:
        bg_df = pd.DataFrame(background_points)
        log("INFO", "BUILD", f"added {len(bg_df)} background negative points")
        points_df = pd.concat([points_df, bg_df], ignore_index=True)

    return points_df


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
        n_background = len(
            split_points[split_points['is_background'] == True]) if 'is_background' in split_points.columns else 0

        info[split_name] = {
            'n_events': n_events,
            'n_points': n_points,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_background': n_background
        }

    return info


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = {
        'event_id', 'symbol', 'open_time', 'offset', 'y',
        'pump_la_type', 'runup_pct', 'split', 'target',
        'timeframe', 'window_bars', 'warmup_bars', 'is_background'
    }
    return [col for col in df.columns if col not in exclude_cols]


def train_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42,
        use_sample_weights: bool = True,
        neg_before: int = 60
) -> CatBoostClassifier:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

    if use_sample_weights:
        train_weights = compute_sample_weights(
            train_df['offset'].values,
            train_df['y'].values,
            neg_before
        )
        train_pool = Pool(X_train, y_train, weight=train_weights)
    else:
        train_pool = Pool(X_train, y_train)

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
        use_best_model=True,
        auto_class_weights='Balanced'
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


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]


def filter_features_by_event_time(features_df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    valid_events = event_times[event_times['open_time'] < cutoff]['event_id']
    return features_df[features_df['event_id'].isin(valid_events)].copy()


def calibrate_threshold_on_val(
        model,
        features_df: pd.DataFrame,
        feature_columns: list,
        train_end: datetime,
        val_end: datetime,
        signal_rule: str,
        alpha_hitM1: float,
        beta_early: float,
        beta_late: float,
        gamma_miss: float,
        lambda_offset: float,
        hysteresis_delta: float
) -> float:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    val_events = event_times[
        (event_times['open_time'] >= train_end) &
        (event_times['open_time'] < val_end)
        ]['event_id']

    val_df = features_df[
        (features_df['event_id'].isin(val_events)) &
        (features_df['open_time'] >= train_end) &
        (features_df['open_time'] < val_end)
        ].copy()

    if len(val_df) == 0:
        return 0.5

    val_df['split'] = 'val'
    predictions = predict_proba_long(model, val_df, feature_columns)

    event_data = _prepare_event_data(predictions)

    threshold, _ = threshold_sweep_long(
        predictions,
        alpha_hitM1=alpha_hitM1,
        beta_early=beta_early,
        beta_late=beta_late,
        gamma_miss=gamma_miss,
        lambda_offset=lambda_offset,
        signal_rule=signal_rule,
        hysteresis_delta=hysteresis_delta,
        event_data=event_data
    )

    return threshold


def run_build_dataset(args, artifacts: RunArtifacts):
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = parse_date_exclusive(args.end_date) if args.end_date else None

    if args.labels:
        log("INFO", "BUILD", f"loading labels from {args.labels}")
        labels_df = load_labels(args.labels, start_date, end_date)
    else:
        log("INFO", "BUILD", f"exporting labels from ClickHouse")
        labels_df = export_pump_long_labels(args.clickhouse_dsn, start_date, end_date, max_workers=args.build_workers)

    log("INFO", "BUILD", f"loaded {len(labels_df)} labels")

    artifacts.save_labels(labels_df)

    pos_offsets = parse_pos_offsets(args.pos_offsets)
    log("INFO", "BUILD",
        f"building training points neg_before={args.neg_before} neg_after={args.neg_after} pos_offsets={pos_offsets}")
    points_df = build_training_points(
        labels_df,
        neg_before=args.neg_before,
        neg_after=args.neg_after,
        pos_offsets=pos_offsets
    )

    if args.background_negatives > 0:
        points_df = add_background_negatives(
            points_df,
            labels_df,
            num_per_symbol=args.background_negatives,
            min_distance_bars=args.background_distance,
            seed=args.seed
        )

    log("INFO", "BUILD",
        f"training points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    artifacts.save_training_points(points_df)

    log("INFO", "BUILD", f"building features from ClickHouse")
    builder = PumpFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=args.window_bars,
        warmup_bars=args.warmup_bars,
        feature_set=args.feature_set
    )

    unique_times = points_df[['symbol', 'open_time']].drop_duplicates()
    feature_input = unique_times.copy()
    feature_input = feature_input.rename(columns={'open_time': 'event_open_time'})
    feature_input['pump_la_type'] = 'A'
    feature_input['runup_pct'] = 0

    features_df = builder.build(feature_input, max_workers=args.build_workers)

    merge_cols = ['symbol', 'open_time', 'event_id', 'offset', 'y']
    if 'is_background' in points_df.columns:
        merge_cols.append('is_background')

    features_df = features_df.merge(
        points_df[merge_cols],
        on=['symbol', 'open_time'],
        how='inner'
    )

    features_df = features_df.sort_values(['event_id', 'offset']).reset_index(drop=True)

    log("INFO", "BUILD", f"features shape: {features_df.shape}")
    artifacts.save_features(features_df)

    log("INFO", "BUILD", f"dataset saved to {artifacts.get_path()}")


def run_train(args, artifacts: RunArtifacts):
    log("INFO", "TRAIN", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TRAIN", f"loaded {len(features_df)} rows with {len(feature_columns)} features")

    train_end = parse_date_exclusive(args.train_end)
    val_end = parse_date_exclusive(args.val_end)

    features_df = time_split(features_df, train_end, val_end)

    if args.embargo_bars > 0:
        features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

    features_df = clip_points_to_split_bounds(features_df, train_end, val_end)

    split_info = get_split_info(features_df)
    artifacts.save_splits(split_info)
    log("INFO", "TRAIN",
        f"split info: train={split_info['train']['n_events']} val={split_info['val']['n_events']} test={split_info['test']['n_events']} events")

    model = train_model(
        features_df,
        feature_columns,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        thread_count=args.thread_count,
        seed=args.seed,
        use_sample_weights=args.use_sample_weights,
        neg_before=args.neg_before
    )

    artifacts.save_model(model)
    log("INFO", "TRAIN", f"model saved")

    importance_df = get_feature_importance(model, feature_columns)
    artifacts.save_feature_importance(importance_df)

    log("INFO", "TRAIN", "predicting on val set")
    val_predictions = predict_proba_long(
        model,
        features_df[features_df['split'] == 'val'],
        feature_columns
    )
    artifacts.save_predictions(val_predictions, 'val')

    val_event_data = _prepare_event_data(val_predictions)

    log("INFO", "TRAIN", "searching optimal threshold")
    best_threshold, sweep_df = threshold_sweep_long(
        val_predictions,
        alpha_hitM1=args.alpha_hitM1,
        beta_early=args.beta_early,
        beta_late=args.beta_late,
        gamma_miss=args.gamma_miss,
        lambda_offset=args.lambda_offset,
        signal_rule=args.signal_rule,
        hysteresis_delta=args.hysteresis_delta,
        event_data=val_event_data
    )

    artifacts.save_threshold_sweep(sweep_df)
    artifacts.save_best_threshold(best_threshold, args.signal_rule, args.hysteresis_delta)
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    log("INFO", "TRAIN", "evaluating on val set")
    val_metrics = evaluate_long(val_predictions, best_threshold, signal_rule=args.signal_rule,
                                hysteresis_delta=args.hysteresis_delta, event_data=val_event_data)
    artifacts.save_metrics(val_metrics, 'val')
    log("INFO", "TRAIN",
        f"val metrics: hit0={val_metrics['event_level']['hit0_rate']:.3f} hitM1={val_metrics['event_level']['hitM1_rate']:.3f} late={val_metrics['event_level']['late_rate']:.3f} miss={val_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "predicting on test set")
    test_predictions = predict_proba_long(
        model,
        features_df[features_df['split'] == 'test'],
        feature_columns
    )
    artifacts.save_predictions(test_predictions, 'test')

    test_event_data = _prepare_event_data(test_predictions)

    log("INFO", "TRAIN", "evaluating on test set")
    test_metrics = evaluate_long(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                 hysteresis_delta=args.hysteresis_delta, event_data=test_event_data)
    artifacts.save_metrics(test_metrics, 'test')
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} hitM1={test_metrics['event_level']['hitM1_rate']:.3f} late={test_metrics['event_level']['late_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "extracting holdout signals")
    signals_df = extract_signals_long(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                      hysteresis_delta=args.hysteresis_delta)
    artifacts.save_predicted_signals(signals_df)
    log("INFO", "TRAIN", f"saved {len(signals_df)} predicted signals")

    log("INFO", "TRAIN", f"done. artifacts saved to {artifacts.get_path()}")


def run_tune(args, artifacts: RunArtifacts):
    log("INFO", "TUNE", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TUNE", f"loaded {len(features_df)} rows with {len(feature_columns)} features")

    train_end = parse_date_exclusive(args.train_end) if args.train_end else None
    val_end = parse_date_exclusive(args.val_end) if args.val_end else None

    if train_end:
        cv_features_df = filter_features_by_event_time(features_df, train_end)
        log("INFO", "TUNE", f"filtered CV data: {len(cv_features_df)} rows (events before {args.train_end})")
    else:
        cv_features_df = features_df

    log("INFO", "TUNE", f"starting tuning with time_budget={args.time_budget_min}min signal_rule={args.signal_rule}")

    tune_result = tune_model_long(
        cv_features_df,
        feature_columns,
        time_budget_min=args.time_budget_min,
        fold_months=args.fold_months,
        min_train_months=args.min_train_months,
        signal_rule=args.signal_rule,
        alpha_hitM1=args.alpha_hitM1,
        beta_early=args.beta_early,
        beta_late=args.beta_late,
        gamma_miss=args.gamma_miss,
        lambda_offset=args.lambda_offset,
        embargo_bars=args.embargo_bars,
        iterations=args.iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        thread_count=args.thread_count,
        seed=args.seed,
        use_sample_weights=args.use_sample_weights,
        neg_before=args.neg_before,
        hysteresis_delta=args.hysteresis_delta
    )

    log("INFO", "TUNE",
        f"tuning completed: {tune_result['trials_completed']} trials in {tune_result['time_elapsed_sec']:.1f}s")
    log("INFO", "TUNE", f"best score: {tune_result['best_score']:.4f}")
    log("INFO", "TUNE", f"best params: {tune_result['best_params']}")

    artifacts.save_best_params(tune_result['best_params'])
    artifacts.save_leaderboard(tune_result['leaderboard'])
    artifacts.save_cv_report(tune_result['best_cv_result'])
    artifacts.save_folds(tune_result['folds'])

    if train_end:
        log("INFO", "TUNE", f"training final model on data up to {args.train_end}")

        final_model = train_final_model_long(
            features_df,
            feature_columns,
            tune_result['best_params'],
            train_end,
            iterations=args.iterations,
            thread_count=args.thread_count,
            seed=args.seed,
            use_sample_weights=args.use_sample_weights,
            neg_before=args.neg_before
        )

        artifacts.save_model(final_model)
        log("INFO", "TUNE", f"final model saved")

        importance_df = get_feature_importance(final_model, feature_columns)
        artifacts.save_feature_importance(importance_df)

        if val_end:
            log("INFO", "TUNE", f"calibrating threshold on val window [{args.train_end}, {args.val_end})")

            best_threshold = calibrate_threshold_on_val(
                final_model,
                features_df,
                feature_columns,
                train_end,
                val_end,
                args.signal_rule,
                args.alpha_hitM1,
                args.beta_early,
                args.beta_late,
                args.gamma_miss,
                args.lambda_offset,
                args.hysteresis_delta
            )

            log("INFO", "TUNE", f"calibrated threshold: {best_threshold:.3f}")
            artifacts.save_best_threshold(best_threshold, args.signal_rule, args.hysteresis_delta)

            features_df = time_split(features_df, train_end, val_end)

            if args.embargo_bars > 0:
                features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

            features_df = clip_points_to_split_bounds(features_df, train_end, val_end)

            test_predictions = predict_proba_long(
                final_model,
                features_df[features_df['split'] == 'test'],
                feature_columns
            )
            artifacts.save_predictions(test_predictions, 'test')

            test_event_data = _prepare_event_data(test_predictions)

            test_metrics = evaluate_long(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                         hysteresis_delta=args.hysteresis_delta, event_data=test_event_data)
            artifacts.save_metrics(test_metrics, 'test')
            log("INFO", "TUNE",
                f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} hitM1={test_metrics['event_level']['hitM1_rate']:.3f} late={test_metrics['event_level']['late_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

            signals_df = extract_signals_long(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                              hysteresis_delta=args.hysteresis_delta)
            artifacts.save_predicted_signals(signals_df)
            log("INFO", "TUNE", f"saved {len(signals_df)} predicted signals")

    log("INFO", "TUNE", f"done. artifacts saved to {artifacts.get_path()}")


def main():
    parser = argparse.ArgumentParser(description="Train pump long prediction model")

    parser.add_argument("--mode", type=str, choices=["build-dataset", "train", "tune"], required=True)

    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--dataset-parquet", type=str, default=None)

    parser.add_argument("--neg-before", type=int, default=60)
    parser.add_argument("--neg-after", type=int, default=10)
    parser.add_argument("--pos-offsets", type=str, default="0")
    parser.add_argument("--background-negatives", type=int, default=30)
    parser.add_argument("--background-distance", type=int, default=300)

    parser.add_argument("--window-bars", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=150)
    parser.add_argument("--feature-set", type=str, choices=["base", "extended"], default="extended")
    parser.add_argument("--build-workers", type=int, default=4)

    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--embargo-bars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--thread-count", type=int, default=-1)

    parser.add_argument("--signal-rule", type=str,
                        choices=["first_cross", "cross_up", "hysteresis", "pending_turn_up"],
                        default="cross_up")
    parser.add_argument("--alpha-hitM1", type=float, default=0.8)
    parser.add_argument("--beta-early", type=float, default=5.0)
    parser.add_argument("--beta-late", type=float, default=3.0)
    parser.add_argument("--gamma-miss", type=float, default=0.3)
    parser.add_argument("--lambda-offset", type=float, default=0.02)
    parser.add_argument("--hysteresis-delta", type=float, default=0.05)

    parser.add_argument("--use-sample-weights", action="store_true", default=True)
    parser.add_argument("--no-sample-weights", action="store_false", dest="use_sample_weights")

    parser.add_argument("--time-budget-min", type=int, default=60)
    parser.add_argument("--fold-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=3)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "build-dataset":
        if not args.clickhouse_dsn:
            parser.error("--clickhouse-dsn required for build-dataset mode")

    if args.mode == "train":
        if not args.dataset_parquet:
            parser.error("--dataset-parquet required for train mode")
        if not args.train_end or not args.val_end:
            parser.error("--train-end and --val-end required for train mode")

    if args.mode == "tune":
        if not args.dataset_parquet:
            parser.error("--dataset-parquet required for tune mode")

    artifacts = RunArtifacts(args.out_dir, args.run_name)
    log("INFO", "MAIN", f"run_dir={artifacts.get_path()}")

    config = vars(args)
    artifacts.save_config(config)

    if args.mode == "build-dataset":
        run_build_dataset(args, artifacts)
    elif args.mode == "train":
        run_train(args, artifacts)
    elif args.mode == "tune":
        run_tune(args, artifacts)


if __name__ == "__main__":
    main()
