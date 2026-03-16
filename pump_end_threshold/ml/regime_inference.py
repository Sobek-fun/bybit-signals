import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
from pump_end_threshold.infra.clickhouse import DataLoader, get_liquid_universe
from pump_end_threshold.infra.logging import log
from pump_end_threshold.ml.regime_dataset import STRATEGY_STATE_MODE, build_strategy_state_live
from pump_end_threshold.ml.regime_policy import RegimePolicy


@dataclass
class GuardArtifacts:
    model: CatBoostClassifier
    policy_params: dict
    feature_columns: list[str]
    liquid_universe: Optional[list[str]]
    top_n_universe: int
    strategy_state_mode: str
    tp_pct: float
    sl_pct: float
    max_horizon_bars: int
    trade_replay_source: Optional[str]


def load_guard_artifacts(guard_model_dir: Path, ch_dsn: str) -> GuardArtifacts:
    guard_model_path = guard_model_dir / "regime_guard_model.cbm"
    guard_model = CatBoostClassifier()
    guard_model.load_model(str(guard_model_path))

    policy_path = guard_model_dir / "resolved_policy_params.json"
    if not policy_path.exists():
        policy_path = guard_model_dir / "best_policy_params.json"
    with open(policy_path, 'r') as f:
        policy_params = json.load(f)

    with open(guard_model_dir / "feature_columns.json", 'r') as f:
        guard_feature_columns = json.load(f)

    liquid_universe_path = guard_model_dir / "liquid_universe.json"
    if liquid_universe_path.exists():
        log("INFO", "GUARD", f"loading liquid_universe from {liquid_universe_path}")
        with open(liquid_universe_path, 'r') as f:
            liquid_universe = json.load(f)
    else:
        liquid_universe = None

    regime_config_path = guard_model_dir / "regime_builder_config.json"
    if regime_config_path.exists():
        log("INFO", "GUARD", f"loading regime builder config from {regime_config_path}")
        with open(regime_config_path, 'r') as f:
            regime_config = json.load(f)
        top_n = regime_config.get('top_n_universe', 120)
        strategy_state_mode = regime_config.get('strategy_state_mode', STRATEGY_STATE_MODE)
        trade_replay_source = regime_config.get('trade_replay_source')
    else:
        top_n = 120
        strategy_state_mode = STRATEGY_STATE_MODE
        trade_replay_source = None

    if strategy_state_mode != STRATEGY_STATE_MODE:
        raise ValueError(
            f"Unsupported strategy_state_mode={strategy_state_mode}, expected {STRATEGY_STATE_MODE}"
        )

    tp_pct = 4.5
    sl_pct = 10.0
    max_horizon_bars = 200
    manifest_path = guard_model_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        tp_pct = manifest.get('tp_pct', tp_pct)
        sl_pct = manifest.get('sl_pct', sl_pct)
        max_horizon_bars = manifest.get('max_horizon_bars', max_horizon_bars)
        if trade_replay_source is None:
            trade_replay_source = manifest.get('trade_replay_source')

    return GuardArtifacts(
        model=guard_model,
        policy_params=policy_params,
        feature_columns=guard_feature_columns,
        liquid_universe=liquid_universe,
        top_n_universe=top_n,
        strategy_state_mode=strategy_state_mode,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_horizon_bars=max_horizon_bars,
        trade_replay_source=trade_replay_source,
    )


def apply_guard_to_raw_signals(
        raw_signals_df: pd.DataFrame,
        guard_model_dir: Path,
        ch_dsn: str,
        run_dir: Path = None,
        guard_debug_output: str = None,
        blocked_signals_output: str = None,
        accepted_signals_output: str = None,
        artifacts: GuardArtifacts = None,
) -> pd.DataFrame:
    if artifacts is None:
        artifacts = load_guard_artifacts(guard_model_dir, ch_dsn)
    if raw_signals_df.empty:
        return raw_signals_df

    signals = raw_signals_df.sort_values('open_time').reset_index(drop=True).copy()
    t_min = signals['open_time'].min()

    liquid_universe = artifacts.liquid_universe
    if not liquid_universe:
        liquid_universe = get_liquid_universe(
            ch_dsn, t_min - timedelta(days=7), t_min, top_n=artifacts.top_n_universe
        )

    builder = RegimeFeatureBuilder(
        ch_dsn=ch_dsn,
        liquid_universe=liquid_universe,
        top_n=artifacts.top_n_universe,
    )

    loader = DataLoader(ch_dsn)
    log("INFO", "GUARD", "building causal strategy-state snapshots")
    strategy_state_by_time = {}
    unique_times = pd.Series(signals['open_time'].dropna().unique()).sort_values()
    replay_source = artifacts.trade_replay_source or "1s"
    for t in unique_times:
        asof_time = pd.Timestamp(t).to_pydatetime()
        strategy_state_by_time[pd.Timestamp(t)] = build_strategy_state_live(
            signals,
            loader,
            asof_time=asof_time,
            tp_pct=artifacts.tp_pct,
            sl_pct=artifacts.sl_pct,
            max_horizon_bars=artifacts.max_horizon_bars,
            trade_replay_source=replay_source,
        )

    log("INFO", "GUARD", "building regime features in batch mode")
    guard_features = builder.build_batch(signals, batch_size=100, trades_df=strategy_state_by_time)

    if guard_features.empty:
        log("WARN", "GUARD", "no features built")
        return raw_signals_df

    available = [c for c in artifacts.feature_columns if c in guard_features.columns]
    if len(available) < len(artifacts.feature_columns):
        missing = set(artifacts.feature_columns) - set(available)
        log("WARN", "GUARD", f"missing {len(missing)} feature columns, filling with NaN: {list(missing)[:10]}")
        for col in missing:
            guard_features[col] = np.nan

    X_guard = guard_features[artifacts.feature_columns]
    p_bad = artifacts.model.predict_proba(X_guard)[:, 1]
    signals['p_bad'] = p_bad

    if guard_debug_output:
        signals.to_parquet(guard_debug_output, index=False)
    elif run_dir:
        signals.to_parquet(run_dir / "guard_scored_signals.parquet", index=False)

    if 'pause_on_quantile' in artifacts.policy_params:
        raise ValueError(
            "Quantile policy is not allowed in export. Provide numeric thresholds in resolved_policy_params.json"
        )

    policy = RegimePolicy(**artifacts.policy_params)
    result = policy.apply(signals, p_bad_col='p_bad')

    accepted = result[~result['blocked_by_policy']].copy()
    blocked = result[result['blocked_by_policy']].copy()

    if accepted_signals_output:
        accepted.to_parquet(accepted_signals_output, index=False)
    elif run_dir:
        accepted.to_parquet(run_dir / "accepted_signals.parquet", index=False)

    if blocked_signals_output:
        blocked.to_parquet(blocked_signals_output, index=False)
    elif run_dir:
        blocked.to_parquet(run_dir / "blocked_signals.parquet", index=False)

    n_blocked = int(result['blocked_by_policy'].sum())
    log("INFO", "GUARD", f"guard applied: {len(result)} raw -> {len(accepted)} accepted ({n_blocked} blocked)")
    return accepted
