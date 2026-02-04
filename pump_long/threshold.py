import numpy as np
import pandas as pd


def _prepare_event_data(predictions_df: pd.DataFrame) -> dict:
    event_data = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        event_data[event_id] = {
            'offsets': sorted_group['offset'].values,
            'p_long': sorted_group['p_long'].values
        }
    return event_data


def _compute_event_metrics_long(
        event_data: dict,
        threshold: float,
        signal_rule: str = 'first_cross'
) -> dict:
    hit0 = 0
    hitM1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []

    for event_id, data in event_data.items():
        if signal_rule == 'first_cross':
            mask = data['p_long'] >= threshold
            if not mask.any():
                miss += 1
                continue
            first_idx = np.argmax(mask)
            offset = data['offsets'][first_idx]
        else:
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
        'threshold': threshold,
        'n_events': n_events,
        'hit0': hit0,
        'hitM1': hitM1,
        'early': early,
        'late': late,
        'miss': miss,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hitM1_rate': hitM1 / n_events if n_events > 0 else 0,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_offset': np.mean(offsets) if offsets else None,
        'median_offset': np.median(offsets) if offsets else None
    }


def threshold_sweep_long(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.01,
        grid_to: float = 0.50,
        grid_step: float = 0.01,
        alpha_hitM1: float = 0.8,
        beta_early: float = 1.0,
        beta_late: float = 3.0,
        gamma_miss: float = 1.0,
        signal_rule: str = 'first_cross',
        event_data: dict = None
) -> tuple:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = _compute_event_metrics_long(event_data, threshold, signal_rule)

        score = (
                metrics['hit0_rate'] +
                alpha_hitM1 * metrics['hitM1_rate'] -
                beta_early * metrics['early_rate'] -
                beta_late * metrics['late_rate'] -
                gamma_miss * metrics['miss_rate']
        )
        metrics['score'] = score

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
