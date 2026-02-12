import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pump_end.ml.threshold import _prepare_event_data


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


def extract_signals(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        abstain_margin: float = 0.0
) -> pd.DataFrame:
    event_data = _prepare_event_data(predictions_df)

    threshold_high = threshold
    threshold_low = max(0.0, threshold - abstain_margin)

    symbol_map = predictions_df.groupby('event_id')['symbol'].first().to_dict()
    time_map = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        time_map[event_id] = dict(zip(sorted_group['offset'], sorted_group['open_time']))

    signals = []

    for event_id, data in event_data.items():
        offsets_arr = data['offsets']
        p_end = data['p_end']
        event_type = data.get('event_type', 'A')

        triggered = False
        pending_count = 0

        for i in range(len(offsets_arr)):
            if p_end[i] >= threshold_high:
                pending_count += 1
                if pending_count >= min_pending_bars and i > 0:
                    drop = p_end[i - 1] - p_end[i]
                    if p_end[i] < p_end[i - 1] and drop >= drop_delta:
                        offset = offsets_arr[i]
                        triggered = True
                        break
            elif p_end[i] < threshold_low:
                pending_count = 0

        if not triggered:
            continue

        signals.append({
            'symbol': symbol_map[event_id],
            'open_time': time_map[event_id][offset],
            'event_type': event_type
        })

    return pd.DataFrame(signals)
