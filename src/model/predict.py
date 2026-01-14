import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.model.threshold import _prepare_event_data


def predict_proba(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list
) -> pd.DataFrame:
    X = features_df[feature_columns]
    proba = model.predict_proba(X)[:, 1]

    result_df = features_df[['event_id', 'symbol', 'close_time', 'offset', 'y', 'split']].copy()
    result_df['p_end'] = proba

    return result_df


def extract_signals(
        predictions_df: pd.DataFrame,
        threshold: float
) -> pd.DataFrame:
    event_data = _prepare_event_data(predictions_df)

    symbol_map = predictions_df.groupby('event_id')['symbol'].first().to_dict()
    time_map = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        time_map[event_id] = dict(zip(sorted_group['offset'], sorted_group['close_time']))

    signals = []

    for event_id, data in event_data.items():
        mask = data['p_end'] >= threshold
        if not mask.any():
            continue

        first_idx = np.argmax(mask)
        offset = data['offsets'][first_idx]

        signals.append({
            'symbol': symbol_map[event_id],
            'close_time': time_map[event_id][offset]
        })

    return pd.DataFrame(signals)
