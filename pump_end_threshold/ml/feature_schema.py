def prune_feature_columns(feature_columns: list) -> list:
    prune_prefixes = [
        'liq_sweep_flag_lag_',
        'liq_sweep_overshoot_lag_',
        'liq_reject_strength_lag_',
    ]

    prune_names = {
        'touched_pdh', 'touched_pwh',
        'sweep_pdh', 'sweep_pwh', 'sweep_eqh',
        'overshoot_pdh', 'overshoot_pwh', 'overshoot_eqh',
        'liq_level_type_pwh',
        'vol_spike_cond', 'vol_spike_recent',
        'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'osc_extreme', 'macd_pos_recent',
        'pump_ctx', 'strong_cond', 'pump_score',
        'predump_mask', 'predump_peak',
    }

    pruned = []
    for col in feature_columns:
        if col in prune_names:
            continue
        if any(col.startswith(prefix) for prefix in prune_prefixes):
            continue
        pruned.append(col)

    return pruned
