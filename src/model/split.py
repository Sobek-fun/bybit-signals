from datetime import datetime

import numpy as np
import pandas as pd


def time_split(
        points_df: pd.DataFrame,
        train_end: datetime,
        val_end: datetime
) -> pd.DataFrame:
    event_times = points_df[points_df['offset'] == 0][['event_id', 'close_time']].copy()
    event_times = event_times.drop_duplicates('event_id')

    event_to_split = {}
    for _, row in event_times.iterrows():
        event_id = row['event_id']
        event_time = row['close_time']

        if event_time <= train_end:
            event_to_split[event_id] = 'train'
        elif event_time <= val_end:
            event_to_split[event_id] = 'val'
        else:
            event_to_split[event_id] = 'test'

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

    event_to_split = {}
    for i, event_id in enumerate(event_ids):
        if i < train_end_idx:
            event_to_split[event_id] = 'train'
        elif i < val_end_idx:
            event_to_split[event_id] = 'val'
        else:
            event_to_split[event_id] = 'test'

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
