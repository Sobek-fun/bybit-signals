import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_long.threshold import threshold_sweep_long, _prepare_event_data
from pump_long.tuning import predict_proba_long


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

    def save_predicted_signals_event_windows(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_event_windows.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals_holdout(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_holdout.csv"
        df.to_csv(path, index=False)

    def save_stream_metrics(self, metrics: dict):
        path = self.run_dir / "stream_metrics_holdout.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_predictions(self, df: pd.DataFrame, split_name: str):
        path = self.run_dir / f"predictions_{split_name}.parquet"
        df.to_parquet(path, index=False)

    def save_best_params(self, params: dict):
        path = self.run_dir / "best_params.json"
        with open(path, 'w') as f:
            json.dump(params, f, indent=2, default=str)

    def save_best_threshold(self, threshold: float, signal_rule: str = 'first_cross'):
        data = {'threshold': threshold, 'signal_rule': signal_rule}
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


def build_training_points_from_labels(
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
        'runup_pct': runup_pcts
    })

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

        info[split_name] = {
            'n_events': n_events,
            'n_points': n_points,
            'n_positive': n_positive,
            'n_negative': n_negative
        }

    return info


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = {
        'event_id', 'symbol', 'open_time', 'offset', 'y',
        'pump_la_type', 'runup_pct', 'split', 'target',
        'timeframe', 'window_bars', 'warmup_bars'
    }
    return [col for col in df.columns if col not in exclude_cols]


def train_catboost_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42
) -> CatBoostClassifier:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

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
        gamma_miss: float
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
        return 0.1

    val_df['split'] = 'val'
    predictions = predict_proba_long(model, val_df, feature_columns)

    event_data = _prepare_event_data(predictions)

    threshold, _ = threshold_sweep_long(
        predictions,
        alpha_hitM1=alpha_hitM1,
        beta_early=beta_early,
        beta_late=beta_late,
        gamma_miss=gamma_miss,
        signal_rule=signal_rule,
        event_data=event_data
    )

    return threshold


def filter_features_by_event_time(features_df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    valid_events = event_times[event_times['open_time'] < cutoff]['event_id']
    return features_df[features_df['event_id'].isin(valid_events)].copy()


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]
