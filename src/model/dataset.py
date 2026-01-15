from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def load_labels(path: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'event_open_time' not in df.columns:
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'event_open_time'})
        elif 'close_time' in df.columns:
            df['event_open_time'] = pd.to_datetime(df['close_time'], utc=True).dt.tz_localize(None) - timedelta(
                minutes=15)
            df = df.drop(columns=['close_time'])

    df['event_open_time'] = pd.to_datetime(df['event_open_time'], utc=True).dt.tz_localize(None)

    if start_date:
        df = df[df['event_open_time'] >= start_date]
    if end_date:
        df = df[df['event_open_time'] < end_date]

    return df


def build_training_points(
        labels_df: pd.DataFrame,
        neg_before: int = 20,
        neg_after: int = 0,
        pos_offsets: list = None,
        include_b: bool = False
) -> pd.DataFrame:
    if pos_offsets is None:
        pos_offsets = [0]

    if include_b:
        events = labels_df[labels_df['pump_la_type'].isin(['A', 'B'])].copy()
    else:
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
    pump_types = np.repeat(events['pump_la_type'].values, n_offsets)
    runup_pcts = np.repeat(events['runup_pct'].values if 'runup_pct' in events.columns else np.nan, n_offsets)

    offsets = np.tile(all_offsets, n_events)
    time_deltas = pd.to_timedelta(offsets * 15, unit='m')
    open_times = pd.to_datetime(base_times) + time_deltas

    pos_offsets_set = set(pos_offsets)
    y_values = np.where(
        np.isin(offsets, list(pos_offsets_set)) & (np.tile(events['pump_la_type'].values, n_offsets) == 'A'),
        1,
        0
    )

    points_df = pd.DataFrame({
        'event_id': event_ids,
        'symbol': symbols,
        'open_time': open_times,
        'offset': offsets,
        'y': y_values,
        'pump_la_type': pump_types,
        'runup_pct': runup_pcts
    })

    return points_df


def deduplicate_points(points_df: pd.DataFrame) -> pd.DataFrame:
    points_df = points_df.sort_values('y', ascending=False)
    points_df = points_df.drop_duplicates(subset=['symbol', 'open_time'], keep='first')
    points_df = points_df.sort_values(['symbol', 'open_time']).reset_index(drop=True)
    return points_df
