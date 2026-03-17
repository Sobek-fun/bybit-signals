from datetime import datetime, timedelta

import numpy as np
import pandas as pd


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


def ratio_split(
        points_df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42
) -> pd.DataFrame:
    event_ids = points_df['event_id'].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(event_ids)

    n = len(event_ids)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    splits = np.empty(n, dtype=object)
    splits[:train_end_idx] = 'train'
    splits[train_end_idx:val_end_idx] = 'val'
    splits[val_end_idx:] = 'test'

    event_to_split = dict(zip(event_ids, splits))

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

    split_col = points_df['split']
    open_time_col = points_df['open_time']

    train_mask = split_col == 'train'
    val_mask = split_col == 'val'
    test_mask = split_col == 'test'

    keep_mask = pd.Series(True, index=points_df.index)

    keep_mask &= ~(train_mask & (open_time_col >= train_end))
    keep_mask &= ~(val_mask & (open_time_col < train_end))
    keep_mask &= ~(val_mask & (open_time_col >= val_end))
    keep_mask &= ~(test_mask & (open_time_col < val_end))

    if test_end:
        keep_mask &= ~(test_mask & (open_time_col >= test_end))

    return points_df[keep_mask].reset_index(drop=True)


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
