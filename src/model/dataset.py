from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def load_labels(path: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'timestamp' in df.columns and 'close_time' not in df.columns:
        df = df.rename(columns={'timestamp': 'close_time'})

    df['close_time'] = pd.to_datetime(df['close_time'], utc=True).dt.tz_localize(None)

    if start_date:
        df = df[df['close_time'] >= start_date]
    if end_date:
        df = df[df['close_time'] < end_date]

    return df


def build_training_points(
        labels_df: pd.DataFrame,
        neg_before: int = 20,
        neg_after: int = 0,
        include_b: bool = False
) -> pd.DataFrame:
    if include_b:
        events = labels_df[labels_df['pump_la_type'].isin(['A', 'B'])].copy()
    else:
        events = labels_df[labels_df['pump_la_type'] == 'A'].copy()

    if events.empty:
        return pd.DataFrame()

    events['event_id'] = events['symbol'] + '|' + events['close_time'].dt.strftime('%Y%m%d_%H%M%S')

    all_offsets = [0] + list(range(-neg_before, 0)) + list(range(1, neg_after + 1))
    n_events = len(events)
    n_offsets = len(all_offsets)

    event_ids = np.repeat(events['event_id'].values, n_offsets)
    symbols = np.repeat(events['symbol'].values, n_offsets)
    base_times = np.repeat(events['close_time'].values, n_offsets)
    pump_types = np.repeat(events['pump_la_type'].values, n_offsets)
    runup_pcts = np.repeat(events['runup_pct'].values if 'runup_pct' in events.columns else np.nan, n_offsets)

    offsets = np.tile(all_offsets, n_events)
    time_deltas = pd.to_timedelta(offsets * 15, unit='m')
    close_times = pd.to_datetime(base_times) + time_deltas

    y_values = np.where(
        (offsets == 0) & (pump_types == 'A'),
        1,
        0
    )

    points_df = pd.DataFrame({
        'event_id': event_ids,
        'symbol': symbols,
        'close_time': close_times,
        'offset': offsets,
        'y': y_values,
        'pump_la_type': pump_types,
        'runup_pct': runup_pcts
    })

    return points_df


def deduplicate_points(points_df: pd.DataFrame) -> pd.DataFrame:
    points_df = points_df.sort_values('y', ascending=False)
    points_df = points_df.drop_duplicates(subset=['symbol', 'close_time'], keep='first')
    points_df = points_df.sort_values(['symbol', 'close_time']).reset_index(drop=True)
    return points_df
