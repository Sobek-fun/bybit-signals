import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)

from src.model.threshold import find_first_signal_offset


def compute_event_level_metrics(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    event_ids = predictions_df['event_id'].unique()

    hit0 = 0
    hit1 = 0
    early = 0
    late = 0
    miss = 0

    offsets = []

    for event_id in event_ids:
        event_df = predictions_df[predictions_df['event_id'] == event_id]
        offset = find_first_signal_offset(event_df, threshold)

        if offset is None:
            miss += 1
        elif offset == 0:
            hit0 += 1
            offsets.append(offset)
        elif offset == 1:
            hit1 += 1
            offsets.append(offset)
        elif offset < 0:
            early += 1
            offsets.append(offset)
        else:
            late += 1
            offsets.append(offset)

    n_events = len(event_ids)

    return {
        'n_events': n_events,
        'hit0': hit0,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hit0_or_hit1': hit0 + hit1,
        'hit0_or_hit1_rate': (hit0 + hit1) / n_events if n_events > 0 else 0,
        'early': early,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late': late,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss': miss,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_pred_offset': np.mean(offsets) if offsets else None,
        'median_pred_offset': np.median(offsets) if offsets else None
    }


def compute_point_level_metrics(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    y_true = predictions_df['y'].values
    y_prob = predictions_df['p_end'].values
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


def evaluate(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    event_metrics = compute_event_level_metrics(predictions_df, threshold)
    point_metrics = compute_point_level_metrics(predictions_df, threshold)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics
    }
