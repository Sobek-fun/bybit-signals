import pandas as pd


EXCLUDED_COLUMNS = {
    'event_id',
    'signal_id',
    'symbol',
    'open_time',
    'event_type',
    'signal_offset',
    'p_end_at_fire',
    'p_end_peak_before_fire',
    'threshold_used',
    'threshold_gap',
    'pending_bars',
    'drop_from_peak_at_fire',
    'source_model_run',
    'split',
    'fold_idx',
    'sample_weight',

    'trade_outcome',
    'tp_hit',
    'sl_hit',
    'exit_time',
    'entry_price',
    'exit_price',
    'pnl_pct',
    'mfe_pct',
    'mae_pct',
    'trade_duration_bars',

    'pause_state',
    'blocked_by_policy',
    'accepted_by_policy',
    'p_bad',
    'regime_state',
    'month',
}

EXCLUDED_PREFIXES = (
    'target_',
    'future_',
    'next_',
    'policy_',
    'blocked_',
)


def get_regime_feature_columns(df: pd.DataFrame) -> list:
    features = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if any(col.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        features.append(col)
    return features
