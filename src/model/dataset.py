from datetime import datetime, timedelta

import pandas as pd


def load_labels(path: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'timestamp' in df.columns and 'close_time' not in df.columns:
        df = df.rename(columns={'timestamp': 'close_time'})

    df['close_time'] = pd.to_datetime(df['close_time'], utc=True).dt.tz_localize(None)

    if start_date:
        df = df[df['close_time'] >= start_date]
    if end_date:
        df = df[df['close_time'] <= end_date]

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

    points = []

    for _, row in events.iterrows():
        symbol = row['symbol']
        close_time = row['close_time']
        event_id = f"{symbol}|{close_time.strftime('%Y%m%d_%H%M%S')}"

        points.append({
            'event_id': event_id,
            'symbol': symbol,
            'close_time': close_time,
            'offset': 0,
            'y': 1,
            'pump_la_type': row['pump_la_type'],
            'runup_pct': row.get('runup_pct', None)
        })

        for k in range(1, neg_before + 1):
            neg_time = close_time - timedelta(minutes=15 * k)
            points.append({
                'event_id': event_id,
                'symbol': symbol,
                'close_time': neg_time,
                'offset': -k,
                'y': 0,
                'pump_la_type': row['pump_la_type'],
                'runup_pct': row.get('runup_pct', None)
            })

        for m in range(1, neg_after + 1):
            pos_time = close_time + timedelta(minutes=15 * m)
            points.append({
                'event_id': event_id,
                'symbol': symbol,
                'close_time': pos_time,
                'offset': m,
                'y': 0,
                'pump_la_type': row['pump_la_type'],
                'runup_pct': row.get('runup_pct', None)
            })

    points_df = pd.DataFrame(points)
    return points_df


def deduplicate_points(points_df: pd.DataFrame) -> pd.DataFrame:
    points_df = points_df.sort_values('y', ascending=False)
    points_df = points_df.drop_duplicates(subset=['symbol', 'close_time'], keep='first')
    points_df = points_df.sort_values(['symbol', 'close_time']).reset_index(drop=True)
    return points_df
