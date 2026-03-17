import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

def predict_proba(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list
) -> pd.DataFrame:
    X = features_df[feature_columns]
    proba = model.predict_proba(X)[:, 1]

    keep_cols = ['event_id', 'symbol', 'open_time', 'offset', 'y', 'split']
    if 'pump_la_type' in features_df.columns:
        keep_cols.append('pump_la_type')

    result_df = features_df[keep_cols].copy()
    result_df['p_end'] = proba

    return result_df


def prepare_event_data(predictions_df: pd.DataFrame) -> dict:
    has_type = 'pump_la_type' in predictions_df.columns
    event_data = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        offsets = sorted_group['offset'].to_numpy()
        open_times = sorted_group['open_time'].to_numpy()
        entry = {
            'offsets': offsets,
            'p_end': sorted_group['p_end'].to_numpy(),
            'symbol': sorted_group['symbol'].iloc[0] if len(sorted_group) > 0 else None,
            'offset_to_open_time': dict(zip(offsets, open_times)),
        }
        if has_type:
            types = sorted_group['pump_la_type'].to_numpy()
            entry['event_type'] = types[0] if len(types) > 0 else 'A'
        else:
            entry['event_type'] = 'A'
        event_data[event_id] = entry
    return event_data


def build_pending_turn_down_decision_table(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        abstain_margin: float = 0.0,
        event_data: dict = None
) -> pd.DataFrame:
    if signal_rule != 'pending_turn_down':
        raise ValueError(f"Unsupported signal_rule: {signal_rule}")

    if event_data is None:
        event_data = prepare_event_data(predictions_df)

    threshold_high = threshold
    threshold_low = max(0.0, threshold - abstain_margin)
    rows = []

    for event_id, data in event_data.items():
        offsets_arr = data['offsets']
        p_end = data['p_end']
        event_type = data.get('event_type', 'A')
        symbol = data.get('symbol')
        offset_to_open_time = data.get('offset_to_open_time', {})

        triggered = False
        pending_count = 0
        best_p = -1.0
        best_offset = None
        p_end_peak_before_fire = -1.0
        fire_offset = None
        fire_p_end = None
        drop_from_peak_at_fire = None

        for i in range(len(offsets_arr)):
            if p_end[i] >= threshold_high:
                pending_count += 1
                if p_end[i] > p_end_peak_before_fire:
                    p_end_peak_before_fire = p_end[i]
                if p_end[i] > best_p:
                    best_p = p_end[i]
                    best_offset = offsets_arr[i]

            if pending_count >= min_pending_bars and best_offset is not None:
                drop_from_peak = best_p - p_end[i]
                if p_end[i] < threshold_low or (drop_from_peak > 0 and drop_from_peak >= drop_delta):
                    triggered = True
                    fire_p_end = p_end[i]
                    fire_offset = offsets_arr[i]
                    drop_from_peak_at_fire = p_end_peak_before_fire - fire_p_end
                    break

            if p_end[i] < threshold_low:
                pending_count = 0
                best_p = -1.0
                best_offset = None

        rows.append({
            'event_id': event_id,
            'symbol': symbol,
            'event_type': event_type,
            'triggered': triggered,
            'open_time': offset_to_open_time.get(fire_offset) if triggered else pd.NaT,
            'signal_offset': fire_offset if triggered else np.nan,
            'peak_offset': best_offset if triggered else np.nan,
            'peak_open_time': offset_to_open_time.get(best_offset) if triggered else pd.NaT,
            'p_end_at_fire': fire_p_end if triggered else np.nan,
            'p_end_peak_before_fire': p_end_peak_before_fire if triggered else np.nan,
            'drop_from_peak_at_fire': drop_from_peak_at_fire if triggered else np.nan,
            'threshold_used': threshold,
            'min_pending_bars_used': min_pending_bars,
            'drop_delta_used': drop_delta,
            'abstain_margin_used': abstain_margin,
        })

    if not rows:
        return pd.DataFrame(columns=[
            'event_id', 'symbol', 'event_type', 'triggered', 'open_time', 'signal_offset', 'peak_offset',
            'peak_open_time', 'p_end_at_fire', 'p_end_peak_before_fire', 'drop_from_peak_at_fire',
            'threshold_used', 'min_pending_bars_used', 'drop_delta_used', 'abstain_margin_used',
        ])
    return pd.DataFrame(rows)


def extract_signals(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        abstain_margin: float = 0.0
) -> pd.DataFrame:
    verbose_df = extract_signals_verbose(
        predictions_df=predictions_df,
        threshold=threshold,
        signal_rule=signal_rule,
        min_pending_bars=min_pending_bars,
        drop_delta=drop_delta,
        abstain_margin=abstain_margin,
    )
    if verbose_df.empty:
        return pd.DataFrame(columns=['symbol', 'open_time', 'event_type'])
    result = verbose_df[['symbol', 'open_time', 'event_type']].copy()
    result = result.sort_values(['open_time', 'symbol']).reset_index(drop=True)
    return result


def extract_signals_verbose(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        abstain_margin: float = 0.0
) -> pd.DataFrame:
    decision_df = build_pending_turn_down_decision_table(
        predictions_df=predictions_df,
        threshold=threshold,
        signal_rule=signal_rule,
        min_pending_bars=min_pending_bars,
        drop_delta=drop_delta,
        abstain_margin=abstain_margin,
    )
    if decision_df.empty:
        return decision_df
    verbose = decision_df[decision_df['triggered']].copy()
    if verbose.empty:
        return pd.DataFrame(columns=[
            'event_id', 'symbol', 'open_time', 'event_type', 'signal_offset', 'peak_offset', 'peak_open_time',
            'p_end_at_fire', 'p_end_peak_before_fire', 'drop_from_peak_at_fire', 'threshold_used',
            'min_pending_bars_used', 'drop_delta_used', 'abstain_margin_used',
        ])
    verbose = verbose[[
        'event_id', 'symbol', 'open_time', 'event_type', 'signal_offset', 'peak_offset', 'peak_open_time',
        'p_end_at_fire', 'p_end_peak_before_fire', 'drop_from_peak_at_fire', 'threshold_used',
        'min_pending_bars_used', 'drop_delta_used', 'abstain_margin_used',
    ]]
    verbose = verbose.sort_values(['open_time', 'symbol']).reset_index(drop=True)
    return verbose
