import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.dev.ml.threshold import _prepare_event_data


def predict_proba(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list
) -> pd.DataFrame:
    X = features_df[feature_columns]
    proba = model.predict_proba(X)[:, 1]

    result_df = features_df[['event_id', 'symbol', 'open_time', 'offset', 'y', 'split']].copy()
    result_df['p_end'] = proba

    return result_df


def extract_signals(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0
) -> pd.DataFrame:
    event_data = _prepare_event_data(predictions_df)

    symbol_map = predictions_df.groupby('event_id')['symbol'].first().to_dict()
    time_map = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        time_map[event_id] = dict(zip(sorted_group['offset'], sorted_group['open_time']))

    signals = []

    for event_id, data in event_data.items():
        if signal_rule == 'first_cross':
            mask = data['p_end'] >= threshold
            if not mask.any():
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
            max_p_end_pending = 0.0

            for i in range(len(offsets_arr)):
                if p_end[i] >= threshold:
                    if pending_count == 0:
                        max_p_end_pending = p_end[i]
                    else:
                        max_p_end_pending = max(max_p_end_pending, p_end[i])
                    pending_count += 1

                    if pending_count >= min_pending_bars and i > 0:
                        drop_from_max = max_p_end_pending - p_end[i]
                        if drop_from_max >= drop_delta and p_end[i] < p_end[i - 1]:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0
                    max_p_end_pending = 0.0

            if not triggered:
                continue

        signals.append({
            'symbol': symbol_map[event_id],
            'open_time': time_map[event_id][offset]
        })

    return pd.DataFrame(signals)
