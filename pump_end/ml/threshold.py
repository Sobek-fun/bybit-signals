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


HIT_PRE_WINDOW = 2


def _compute_event_metrics_from_data(
        event_data: dict,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1
) -> dict:
    hit0 = 0
    hit1 = 0
    hit_pre = 0
    early = 0
    early_far = 0
    late = 0
    miss = 0
    offsets = []
    early_offsets = []
    early_far_offsets = []

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
            turn_down_count = 0

            for i in range(len(offsets_arr)):
                if p_end[i] >= threshold:
                    pending_count += 1
                    pending_max = max(pending_max, p_end[i])

                    if i > 0 and p_end[i] < p_end[i - 1]:
                        turn_down_count += 1
                    else:
                        turn_down_count = 0

                    if pending_count >= min_pending_bars and pending_max >= min_pending_peak and i > 0:
                        drop_from_peak = pending_max - p_end[i]
                        if drop_from_peak >= drop_delta and turn_down_count >= min_turn_down_bars:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0
                    pending_max = -np.inf
                    turn_down_count = 0

            if not triggered:
                miss += 1
                continue

        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == 1:
            hit1 += 1
        elif -HIT_PRE_WINDOW <= offset <= -1:
            hit_pre += 1
        elif offset < -HIT_PRE_WINDOW:
            early_far += 1
            early_far_offsets.append(offset)
        else:
            late += 1

        if offset < 0:
            early += 1
            early_offsets.append(offset)

    n_events = len(event_data)

    early_magnitude = np.mean([max(0, -o) for o in early_offsets]) if early_offsets else 0.0
    early_far_magnitude = np.mean([max(0, -o - HIT_PRE_WINDOW) for o in early_far_offsets]) if early_far_offsets else 0.0

    hit_window = hit_pre + hit0 + hit1
    triggered = n_events - miss

    offset_minus2_to_plus2_rate = 0.0
    tail_le_minus10_rate = 0.0
    if offsets:
        offsets_arr = np.array(offsets)
        in_range = np.sum((offsets_arr >= -2) & (offsets_arr <= 2))
        tail_early = np.sum(offsets_arr <= -10)
        offset_minus2_to_plus2_rate = in_range / len(offsets_arr)
        tail_le_minus10_rate = tail_early / len(offsets_arr)

    return {
        'threshold': threshold,
        'n_events': n_events,
        'hit0': hit0,
        'hit1': hit1,
        'hit_pre': hit_pre,
        'hit_window': hit_window,
        'early': early,
        'early_far': early_far,
        'late': late,
        'miss': miss,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hit1_rate': hit1 / n_events if n_events > 0 else 0,
        'hit_pre_rate': hit_pre / n_events if n_events > 0 else 0,
        'hit_window_rate': hit_window / n_events if n_events > 0 else 0,
        'early_rate': early / n_events if n_events > 0 else 0,
        'early_far_rate': early_far / n_events if n_events > 0 else 0,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_offset': np.mean(offsets) if offsets else None,
        'median_offset': np.median(offsets) if offsets else None,
        'early_magnitude': early_magnitude,
        'early_far_magnitude': early_far_magnitude,
        'offset_minus2_to_plus2_rate': offset_minus2_to_plus2_rate,
        'tail_le_minus10_rate': tail_le_minus10_rate
    }


def compute_event_metrics_for_threshold(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1,
        event_data: dict = None
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)
    return _compute_event_metrics_from_data(event_data, threshold, signal_rule, min_pending_bars, drop_delta,
                                            min_pending_peak, min_turn_down_bars)


def threshold_sweep(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.01,
        grid_to: float = 0.30,
        grid_step: float = 0.01,
        alpha_hit1: float = 0.5,
        alpha_hit_pre: float = 0.25,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0,
        kappa_early_magnitude: float = 0.03,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1,
        event_data: dict = None,
        min_trigger_rate: float = 0.10,
        max_trigger_rate: float = 1.0
) -> tuple:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        metrics = _compute_event_metrics_from_data(event_data, threshold, signal_rule, min_pending_bars, drop_delta,
                                                   min_pending_peak, min_turn_down_bars)

        trigger_rate = 1 - metrics['miss_rate']
        if trigger_rate < min_trigger_rate:
            score = -np.inf
        elif trigger_rate > max_trigger_rate:
            score = -np.inf
        else:
            score = (
                    metrics['hit0_rate'] +
                    alpha_hit1 * metrics['hit1_rate'] +
                    alpha_hit_pre * metrics['hit_pre_rate'] -
                    beta_early * metrics['early_far_rate'] -
                    gamma_miss * metrics['miss_rate'] -
                    kappa_early_magnitude * metrics['early_far_magnitude']
            )
        metrics['score'] = score
        metrics['trigger_rate'] = trigger_rate

        results.append(metrics)

    results_df = pd.DataFrame(results)

    valid_scores = results_df[results_df['score'] > -np.inf]
    if valid_scores.empty:
        best_fallback_idx = None
        min_distance = np.inf
        target_rate = (min_trigger_rate + max_trigger_rate) / 2

        for idx, row in results_df.iterrows():
            tr = row['trigger_rate']
            if tr < min_trigger_rate:
                distance = min_trigger_rate - tr
            elif tr > max_trigger_rate:
                distance = tr - max_trigger_rate
            else:
                distance = 0

            if distance < min_distance:
                min_distance = distance
                best_fallback_idx = idx

        if best_fallback_idx is not None:
            results_df.loc[best_fallback_idx, 'fallback_reason'] = 'no_threshold_in_corridor'
            best_threshold = results_df.loc[best_fallback_idx, 'threshold']
            return best_threshold, results_df

        fallback_metrics = {
            'threshold': grid_from,
            'n_events': len(event_data),
            'hit0': 0,
            'hit1': 0,
            'early': 0,
            'late': 0,
            'miss': len(event_data),
            'hit0_rate': 0.0,
            'hit1_rate': 0.0,
            'early_rate': 0.0,
            'late_rate': 0.0,
            'miss_rate': 1.0,
            'avg_offset': None,
            'median_offset': None,
            'early_magnitude': 0.0,
            'trigger_rate': 0.0,
            'score': -np.inf,
            'fallback_reason': 'empty_sweep_fallback'
        }
        results_df = pd.DataFrame([fallback_metrics])
        return grid_from, results_df

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
