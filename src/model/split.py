from datetime import datetime

import numpy as np
import pandas as pd


def time_split(
        points_df: pd.DataFrame,
        train_end: datetime,
        val_end: datetime
) -> pd.DataFrame:
    event_times = points_df[points_df['offset'] == 0][['event_id', 'close_time']].drop_duplicates('event_id')

    conditions = [
        event_times['close_time'] < train_end,
        event_times['close_time'] < val_end
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
