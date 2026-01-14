import numpy as np
import pandas as pd


def find_first_signal_offset(event_df: pd.DataFrame, threshold: float) -> int:
    event_df = event_df.sort_values('offset')
    triggered = event_df[event_df['p_end'] >= threshold]

    if triggered.empty:
        return None

    return triggered.iloc[0]['offset']


def _prepare_event_data(predictions_df: pd.DataFrame) -> dict:
    event_data = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        event_data[event_id] = {
            'offsets': sorted_group['offset'].values,
            'p_end': sorted_group['p_end'].values
        }
    return event_data


def _compute_event_metrics_from_data(event_data: dict, threshold: float) -> dict:
    hit0 = 0
    hit1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []

    for event_id, data in event_data.items():
        mask = data['p_end'] >= threshold
        if not mask.any():
            miss += 1
            continue

        first_idx = np.argmax(mask)
        offset = data['offsets'][first_idx]
        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == 1:
            hit1 += 1
        elif offset < 0:
            early += 1
        else:
            late += 1

    n_events = len(event_data)

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


def compute_event_metrics_for_threshold(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    event_data = _prepare_event_data(predictions_df)
    return _compute_event_metrics_from_data(event_data, threshold)


def threshold_sweep(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.05,
        grid_to: float = 0.95,
        grid_step: float = 0.01,
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0
) -> tuple:
    event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = _compute_event_metrics_from_data(event_data, threshold)

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
