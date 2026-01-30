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


def _compute_event_metrics_from_data(
        event_data: dict,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0
) -> dict:
    hit0 = 0
    hit1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []
    early_offsets = []

    for event_id, data in event_data.items():
        if signal_rule == 'first_cross':
            mask = data['p_end'] >= threshold
            if not mask.any():
                miss += 1
                continue
            first_idx = np.argmax(mask)
            offset = data['offsets'][first_idx]
        elif signal_rule == 'argmax_per_event':
            argmax_idx = np.argmax(data['p_end'])
            offset = data['offsets'][argmax_idx]
        else:
            offsets_arr = data['offsets']
            p_end = data['p_end']

            triggered = False
            pending_count = 0
            pending_max = -np.inf

            for i in range(len(offsets_arr)):
                if p_end[i] >= threshold:
                    pending_count += 1
                    pending_max = max(pending_max, p_end[i])

                    if pending_count >= min_pending_bars and i > 0:
                        drop_from_peak = pending_max - p_end[i]
                        if drop_from_peak >= drop_delta and p_end[i] < p_end[i - 1]:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0
                    pending_max = -np.inf

            if not triggered:
                miss += 1
                continue

        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == 1:
            hit1 += 1
        elif offset < 0:
            early += 1
            early_offsets.append(offset)
        else:
            late += 1

    n_events = len(event_data)

    early_magnitude = np.mean([max(0, -o) for o in early_offsets]) if early_offsets else 0.0

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
        'median_offset': np.median(offsets) if offsets else None,
        'early_magnitude': early_magnitude
    }


def compute_event_metrics_for_threshold(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)
    return _compute_event_metrics_from_data(event_data, threshold, signal_rule, min_pending_bars, drop_delta)


def threshold_sweep(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.01,
        grid_to: float = 0.30,
        grid_step: float = 0.01,
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0,
        kappa_early_magnitude: float = 0.03,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None,
        min_trigger_rate: float = 0.10
) -> tuple:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = _compute_event_metrics_from_data(event_data, threshold, signal_rule, min_pending_bars, drop_delta)

        trigger_rate = 1 - metrics['miss_rate']
        if trigger_rate < min_trigger_rate:
            score = -np.inf
        else:
            score = (
                    metrics['hit0_rate'] +
                    alpha_hit1 * metrics['hit1_rate'] -
                    beta_early * metrics['early_rate'] -
                    gamma_miss * metrics['miss_rate'] -
                    kappa_early_magnitude * metrics['early_magnitude']
            )
        metrics['score'] = score

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
