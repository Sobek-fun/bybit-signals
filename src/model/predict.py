import pandas as pd
from catboost import CatBoostClassifier

from src.model.threshold import find_first_signal_offset


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
    event_ids = predictions_df['event_id'].unique()

    signals = []

    for event_id in event_ids:
        event_df = predictions_df[predictions_df['event_id'] == event_id]
        event_df = event_df.sort_values('offset')

        triggered = event_df[event_df['p_end'] >= threshold]

        if not triggered.empty:
            first_signal = triggered.iloc[0]
            signals.append({
                'symbol': first_signal['symbol'],
                'close_time': first_signal['close_time']
            })

    return pd.DataFrame(signals)
