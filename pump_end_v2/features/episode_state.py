import time

import numpy as np
import pandas as pd

from pump_end_v2.logging import log_info, stage_done, stage_start

EPISODE_STATE_COLUMNS: tuple[str, ...] = (
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "episode_age_bars",
    "bars_since_episode_open",
    "episode_open_close",
    "episode_high_so_far",
    "distance_from_episode_high_pct",
    "episode_runup_from_open_pct",
    "episode_extension_from_open_pct",
    "bars_since_episode_high",
    "drawdown_from_episode_high_so_far",
    "high_retest_count",
    "high_persistence_4",
    "episode_pump_context_streak",
)


def build_episode_state_layer(
    token_state_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    decision_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("LAYERS", "EPISODE_STATE")
    if decision_rows_df.empty:
        out = pd.DataFrame(columns=list(EPISODE_STATE_COLUMNS))
        log_info("LAYERS", "episode_state summary rows_total=0 cols_total=16")
        stage_done("LAYERS", "EPISODE_STATE", elapsed_sec=time.perf_counter() - started)
        return out
    market = token_state_df[["symbol", "open_time", "close"]].rename(
        columns={"open_time": "context_bar_open_time", "close": "context_close"}
    )
    if "episode_open_close" not in decision_rows_df.columns:
        raise ValueError("decision_rows_df missing required column: episode_open_close")
    merged = decision_rows_df.copy()
    merged = merged.merge(
        market,
        on=["symbol", "context_bar_open_time"],
        how="left",
    )
    open_close = pd.to_numeric(merged["episode_open_close"], errors="coerce")
    context_close = pd.to_numeric(merged["context_close"], errors="coerce")
    denom = open_close.where(open_close > 0.0)
    merged["episode_runup_from_open_pct"] = (context_close / denom - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )
    merged["episode_extension_from_open_pct"] = (
        pd.to_numeric(merged["episode_high_so_far"], errors="coerce") / denom - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    merged["drawdown_from_episode_high_so_far"] = pd.to_numeric(
        merged["distance_from_episode_high_pct"], errors="coerce"
    )
    merged = merged.sort_values(
        ["episode_id", "context_bar_open_time"], kind="mergesort"
    ).reset_index(drop=True)
    near_high_now = merged["drawdown_from_episode_high_so_far"].fillna(1.0) <= 0.005
    merged["_is_episode_high_now"] = (
        merged["drawdown_from_episode_high_so_far"].fillna(1.0) <= 1e-12
    )
    merged["_bar_idx_in_episode"] = merged.groupby("episode_id", sort=False).cumcount()
    merged["_last_high_idx"] = merged["_bar_idx_in_episode"].where(
        merged["_is_episode_high_now"]
    )
    merged["_last_high_idx"] = (
        merged.groupby("episode_id", sort=False)["_last_high_idx"]
        .ffill()
        .fillna(0)
        .astype(int)
    )
    merged["bars_since_episode_high"] = (
        merged["_bar_idx_in_episode"] - merged["_last_high_idx"]
    )
    merged["high_retest_count"] = (
        near_high_now.astype(int)
        .groupby(merged["episode_id"], sort=False)
        .cumsum()
        .astype(int)
    )
    merged["high_persistence_4"] = (
        near_high_now.astype(float)
        .groupby(merged["episode_id"], sort=False)
        .rolling(window=4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    pump_flag = merged["pump_context_flag"].fillna(False).astype(bool)
    streak_source = pump_flag.groupby(merged["episode_id"], sort=False).cumsum()
    streak_reset = (
        streak_source.where(~pump_flag, 0)
        .groupby(merged["episode_id"], sort=False)
        .cummax()
    )
    merged["episode_pump_context_streak"] = (streak_source - streak_reset).astype(int)
    merged["bars_since_episode_open"] = merged["episode_age_bars"]
    merged = merged.drop(
        columns=["_is_episode_high_now", "_bar_idx_in_episode", "_last_high_idx"],
        errors="ignore",
    )
    episode_state = merged.loc[:, list(EPISODE_STATE_COLUMNS)].copy()
    log_info(
        "LAYERS",
        f"episode_state summary rows_total={len(episode_state)} cols_total={len(episode_state.columns)}",
    )
    stage_done("LAYERS", "EPISODE_STATE", elapsed_sec=time.perf_counter() - started)
    return episode_state
