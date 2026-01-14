import numpy as np
import pandas as pd


def find_first_signal_offset(event_df: pd.DataFrame, threshold: float) -> int:
    event_df = event_df.sort_values('offset')
    triggered = event_df[event_df['p_end'] >= threshold]

    if triggered.empty:
        return None

    return triggered.iloc[0]['offset']


def compute_event_metrics_for_threshold(
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
        'threshold': threshold,
        'n_events': n_events,
        'hit0': hit0,
        'hit1': hit1,
        'early': early,
        'late': late,
        'miss': miss,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hit1_rate': hit1 / n_events if n_events > 0 else 0,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_offset': np.mean(offsets) if offsets else None,
        'median_offset': np.median(offsets) if offsets else None
    }


def threshold_sweep(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.05,
        grid_to: float = 0.95,
        grid_step: float = 0.01,
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0
) -> tuple:
    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = compute_event_metrics_for_threshold(predictions_df, threshold)

        score = (
                metrics['hit0_rate'] +
                alpha_hit1 * metrics['hit1_rate'] -
                beta_early * metrics['early_rate'] -
                gamma_miss * metrics['miss_rate']
        )
        metrics['score'] = score

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
