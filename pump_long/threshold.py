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
        signal_rule: str = 'first_cross',
        hysteresis_delta: float = 0.05
) -> dict:
    hit0 = 0
    hitM1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []

    for event_id, data in event_data.items():
        offsets_arr = data['offsets']
        p_long = data['p_long']

        if signal_rule == 'first_cross':
            mask = p_long >= threshold
            if not mask.any():
                miss += 1
                continue
            first_idx = np.argmax(mask)
            offset = offsets_arr[first_idx]

        elif signal_rule == 'cross_up':
            triggered = False
            for i in range(1, len(offsets_arr)):
                if p_long[i - 1] < threshold and p_long[i] >= threshold:
                    offset = offsets_arr[i]
                    triggered = True
                    break
            if not triggered:
                if p_long[0] >= threshold:
                    offset = offsets_arr[0]
                else:
                    miss += 1
                    continue

        elif signal_rule == 'hysteresis':
            armed = False
            triggered = False
            arm_threshold = threshold - hysteresis_delta

            for i in range(len(offsets_arr)):
                if not armed:
                    if p_long[i] < arm_threshold:
                        armed = True
                else:
                    if p_long[i] >= threshold:
                        offset = offsets_arr[i]
                        triggered = True
                        break

            if not triggered:
                miss += 1
                continue

        elif signal_rule == 'pending_turn_up':
            triggered = False
            pending_count = 0
            min_pending = 2
            up_delta = 0.01

            for i in range(len(offsets_arr)):
                if p_long[i] >= threshold:
                    pending_count += 1
                    if pending_count >= min_pending and i >= min_pending:
                        if p_long[i] - p_long[i - min_pending] >= up_delta:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0

            if not triggered:
                miss += 1
                continue

        else:
            mask = p_long >= threshold
            if not mask.any():
                miss += 1
                continue
            first_idx = np.argmax(mask)
            offset = offsets_arr[first_idx]

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
        grid_from: float = 0.05,
        grid_to: float = 0.95,
        grid_step: float = 0.01,
        alpha_hitM1: float = 0.8,
        beta_early: float = 5.0,
        beta_late: float = 3.0,
        gamma_miss: float = 0.3,
        lambda_offset: float = 0.02,
        signal_rule: str = 'cross_up',
        hysteresis_delta: float = 0.05,
        event_data: dict = None
) -> tuple:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = _compute_event_metrics_long(event_data, threshold, signal_rule, hysteresis_delta)

        median_offset = metrics['median_offset']
        offset_penalty = 0.0
        if median_offset is not None:
            offset_penalty = lambda_offset * abs(median_offset)

        score = (
                metrics['hit0_rate'] +
                alpha_hitM1 * metrics['hitM1_rate'] -
                beta_early * metrics['early_rate'] -
                beta_late * metrics['late_rate'] -
                gamma_miss * metrics['miss_rate'] -
                offset_penalty
        )
        metrics['score'] = score
        metrics['offset_penalty'] = offset_penalty

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
