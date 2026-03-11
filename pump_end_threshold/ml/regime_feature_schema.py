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
    'pump_la_type',
    'timeframe',
    'window_bars',
    'warmup_bars',
    'target',
    'runup_pct',

    'context_time_used',
    'bucket_p_bad',
    'pause_reason',
    'resume_reason',
    'policy_episode_id',

    'bucket_unique_symbols_now',
    'same_symbol_last_24h',
    'signal_density_min_gap_1h',
    'token_near_high_96',
    'token_in_top_decile',
    'token_relative_heat',
    'token_breakout_vs_market',
    'extreme_hot_market',
    'extreme_vol_spike',
    'both_strong',
    'both_weak',
    'btc_strong_eth_weak',
    'btc_weak_eth_strong',
    'btc_strong_x_eth_strong_x_token_near_high',
    'breadth_vol_spike_x_token_near_high',
    'token_overheated_vs_breadth',
    'btc_up_streak',
    'eth_up_streak',
    'btc_vol_expansion',
    'eth_vol_expansion',
}

EXCLUDED_PREFIXES = (
    'target_',
    'future_',
    'next_',
    'policy_',
    'blocked_',
    'det_',
    'snap_',
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
