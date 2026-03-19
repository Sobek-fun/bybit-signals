import numpy as np
import pandas as pd

from pump_end_threshold.ml.predict import prepare_event_data, build_pending_turn_down_decision_table


def _prepare_event_data(predictions_df: pd.DataFrame) -> dict:
    return prepare_event_data(predictions_df)


def _compute_event_metrics_from_data(
        decision_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        abstain_margin: float = 0.0
) -> dict:
    if decision_df.empty:
        return {
            'threshold': threshold,
            'n_events': 0,
            'hit0': 0,
            'hit1': 0,
            'early': 0,
            'late': 0,
            'miss': 0,
            'hit0_rate': 0,
            'hit1_rate': 0,
            'hit0_or_hit1': 0,
            'hit0_or_hit1_rate': 0,
            'early_rate': 0,
            'late_rate': 0,
            'miss_rate': 0,
            'avg_offset': None,
            'median_offset': None,
            'signal_count': 0,
            'n_b': 0,
            'false_positive_b': 0,
            'true_negative_b': 0,
            'fp_b_rate': 0
        }

    b_df = decision_df[decision_df['event_type'] == 'B']
    a_df = decision_df[decision_df['event_type'] != 'B']
    a_triggered = a_df[a_df['triggered']]
    offsets = a_triggered['signal_offset'].dropna().to_numpy()

    hit0 = int((offsets == 0).sum())
    hit1 = int((offsets == 1).sum())
    early = int((offsets < 0).sum())
    late = int((offsets > 1).sum())
    miss = int((~a_df['triggered']).sum())
    false_positive_b = int((b_df['triggered']).sum())
    true_negative_b = int((~b_df['triggered']).sum())
    n_b = int(len(b_df))
    n_a = int(len(a_df))
    n_events = n_a

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
        'hit0_or_hit1': hit0 + hit1,
        'hit0_or_hit1_rate': (hit0 + hit1) / n_events if n_events > 0 else 0,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_offset': float(np.mean(offsets)) if len(offsets) > 0 else None,
        'median_offset': float(np.median(offsets)) if len(offsets) > 0 else None,
        'signal_count': int(len(a_triggered)),
        'n_b': n_b,
        'false_positive_b': false_positive_b,
        'true_negative_b': true_negative_b,
        'fp_b_rate': false_positive_b / n_b if n_b > 0 else 0
    }


def compute_event_metrics_for_threshold(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None,
        abstain_margin: float = 0.0
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)
    decision_df = build_pending_turn_down_decision_table(
        predictions_df=predictions_df,
        threshold=threshold,
        signal_rule=signal_rule,
        min_pending_bars=min_pending_bars,
        drop_delta=drop_delta,
        abstain_margin=abstain_margin,
        event_data=event_data,
    )
    return _compute_event_metrics_from_data(decision_df, threshold, signal_rule, min_pending_bars, drop_delta, abstain_margin)


def threshold_sweep(
        predictions_df: pd.DataFrame,
        grid_from: float = 0.01,
        grid_to: float = 0.30,
        grid_step: float = 0.01,
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0,
        delta_fp_b: float = 3.0,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None,
        abstain_margin: float = 0.0
) -> tuple:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    thresholds = np.arange(grid_from, grid_to + grid_step, grid_step)
    results = []

    for threshold in thresholds:
        decision_df = build_pending_turn_down_decision_table(
            predictions_df=predictions_df,
            threshold=threshold,
            signal_rule=signal_rule,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            abstain_margin=abstain_margin,
            event_data=event_data,
        )
        metrics = _compute_event_metrics_from_data(
            decision_df, threshold, signal_rule, min_pending_bars, drop_delta, abstain_margin
        )

        score = (
                metrics['hit0_rate'] +
                alpha_hit1 * metrics['hit1_rate'] -
                beta_early * metrics['early_rate'] -
                gamma_miss * metrics['miss_rate'] -
                delta_fp_b * metrics['fp_b_rate']
        )
        metrics['score'] = score

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df
