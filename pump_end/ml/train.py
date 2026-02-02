import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = {
        'event_id', 'symbol', 'open_time', 'offset', 'y',
        'pump_la_type', 'runup_pct', 'split', 'target',
        'timeframe', 'window_bars', 'warmup_bars',
        'cluster_id', 'cluster_dist', 'cluster_confidence'
    }
    return [col for col in df.columns if col not in exclude_cols]


def make_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=float)

    w[df["y"] == 1] *= 8.0

    near = df["offset"].between(-5, -1)
    w[near & (df["y"] == 0)] *= 4.0

    after = df["offset"].between(1, 4)
    w[after & (df["y"] == 0)] *= 2.5

    if "pump_la_type" in df.columns:
        is_b = (df["pump_la_type"] == "B")
        w[is_b & (df["offset"] == 0) & (df["y"] == 0)] *= 6.0

    return w


def train_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42
) -> CatBoostClassifier:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

    w_train = make_weights(train_df)
    w_val = make_weights(val_df)

    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val, weight=w_val)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        early_stopping_rounds=early_stopping_rounds,
        thread_count=thread_count,
        random_seed=seed,
        verbose=100,
        eval_metric='Logloss',
        use_best_model=True,
        auto_class_weights='Balanced'
    )

    model.fit(train_pool, eval_set=val_pool)

    return model


def get_feature_importance(model: CatBoostClassifier, feature_columns: list) -> pd.DataFrame:
    importance = model.get_feature_importance()

    df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def get_feature_importance_grouped(importance_df: pd.DataFrame) -> pd.DataFrame:
    def get_group(feature_name: str) -> str:
        prefixes = [
            ('rsi_14', 'RSI'),
            ('mfi_14', 'MFI'),
            ('macdh_12_26_9', 'MACDh'),
            ('macd_line', 'MACD_line'),
            ('macd_signal', 'MACD_signal'),
            ('drawdown', 'drawdown'),
            ('vol_ratio', 'vol_ratio'),
            ('volume', 'volume'),
            ('log_volume', 'volume'),
            ('ret_1', 'returns'),
            ('cum_ret', 'returns'),
            ('ret_accel', 'returns'),
            ('range', 'candle'),
            ('upper_wick', 'candle'),
            ('lower_wick', 'candle'),
            ('body_ratio', 'candle'),
            ('count_red', 'candle'),
            ('signed_body', 'candle'),
            ('climax_vr', 'candle'),
            ('range_over_atr', 'candle'),
            ('atr', 'ATR'),
            ('bb_', 'BB'),
            ('vwap', 'VWAP'),
            ('obv', 'OBV'),
            ('corridor', 'corridor'),
            ('runup', 'pump_structure'),
            ('vol_spike', 'pump_structure'),
            ('pump_ctx', 'pump_structure'),
            ('pump_score', 'pump_structure'),
            ('near_peak', 'pump_structure'),
            ('strong_cond', 'pump_structure'),
            ('rsi_hot', 'oscillator_state'),
            ('mfi_hot', 'oscillator_state'),
            ('osc_hot', 'oscillator_state'),
            ('osc_extreme', 'oscillator_state'),
            ('macd_pos', 'oscillator_state'),
            ('close_pos', 'exhaustion'),
            ('wick_ratio', 'exhaustion'),
            ('blowoff', 'exhaustion'),
            ('predump', 'exhaustion'),
            ('vol_fade', 'fade'),
            ('rsi_fade', 'fade'),
            ('macd_fade', 'fade'),
            ('pdh', 'liquidity'),
            ('pwh', 'liquidity'),
            ('eqh', 'liquidity'),
            ('liq_', 'liquidity'),
            ('dist_to_', 'liquidity'),
            ('sweep_', 'liquidity'),
            ('overshoot_', 'liquidity'),
            ('touched_', 'liquidity'),
        ]

        for prefix, group in prefixes:
            if feature_name.startswith(prefix):
                return group
        return 'other'

    importance_df = importance_df.copy()
    importance_df['group'] = importance_df['feature'].apply(get_group)

    grouped = importance_df.groupby('group')['importance'].agg(['sum', 'mean', 'count'])
    grouped = grouped.sort_values('sum', ascending=False).reset_index()
    grouped.columns = ['group', 'total_importance', 'mean_importance', 'feature_count']

    return grouped
