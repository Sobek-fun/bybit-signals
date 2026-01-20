import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from src.dev.pump_long.threshold import _prepare_event_data


def compute_event_level_metrics_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross',
        event_data: dict = None
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    hit0 = 0
    hitM1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []

    for event_id, data in event_data.items():
        mask = data['p_long'] >= threshold
        if not mask.any():
            miss += 1
            continue
        first_idx = np.argmax(mask)
        offset = data['offsets'][first_idx]

        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == -1:
            hitM1 += 1
        elif offset < -1:
            early += 1
        else:
            late += 1

    n_events = len(event_data)

    return {
        'n_events': n_events,
        'hit0': hit0,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hitM1': hitM1,
        'hitM1_rate': hitM1 / n_events if n_events > 0 else 0,
        'hit0_or_hitM1': hit0 + hitM1,
        'hit0_or_hitM1_rate': (hit0 + hitM1) / n_events if n_events > 0 else 0,
        'early': early,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late': late,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss': miss,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_pred_offset': np.mean(offsets) if offsets else None,
        'median_pred_offset': np.median(offsets) if offsets else None
    }


def compute_point_level_metrics_long(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    y_true = predictions_df['y'].values
    y_prob = predictions_df['p_long'].values
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision_at_threshold': precision,
        'recall_at_threshold': recall,
        'threshold_used': threshold
    }


def evaluate_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross',
        event_data: dict = None
) -> dict:
    event_metrics = compute_event_level_metrics_long(predictions_df, threshold, signal_rule, event_data)
    point_metrics = compute_point_level_metrics_long(predictions_df, threshold)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics
    }


def extract_signals_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross'
) -> pd.DataFrame:
    event_data = _prepare_event_data(predictions_df)

    symbol_map = predictions_df.groupby('event_id')['symbol'].first().to_dict()
    time_map = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        time_map[event_id] = dict(zip(sorted_group['offset'], sorted_group['open_time']))

    signals = []

    for event_id, data in event_data.items():
        mask = data['p_long'] >= threshold
        if not mask.any():
            continue
        first_idx = np.argmax(mask)
        offset = data['offsets'][first_idx]

        signals.append({
            'symbol': symbol_map[event_id],
            'open_time': time_map[event_id][offset]
        })

    return pd.DataFrame(signals)
