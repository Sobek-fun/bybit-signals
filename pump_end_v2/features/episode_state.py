from __future__ import annotations

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
)


def build_episode_state_layer(
    token_state_df: pd.DataFrame, episodes_df: pd.DataFrame, decision_rows_df: pd.DataFrame
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("LAYERS", "EPISODE_STATE")
    if decision_rows_df.empty:
        out = pd.DataFrame(columns=list(EPISODE_STATE_COLUMNS))
        log_info("LAYERS", "episode_state summary rows_total=0 cols_total=10")
        stage_done("LAYERS", "EPISODE_STATE", elapsed_sec=time.perf_counter() - started)
        return out
    market = token_state_df[["symbol", "open_time", "close"]].rename(
        columns={"open_time": "context_bar_open_time", "close": "context_close"}
    )
    merged = decision_rows_df.merge(
        episodes_df[["episode_id", "episode_open_close"]],
        on="episode_id",
        how="left",
    )
    merged = merged.merge(
        market,
        on=["symbol", "context_bar_open_time"],
        how="left",
    )
    open_close = pd.to_numeric(merged["episode_open_close"], errors="coerce")
    context_close = pd.to_numeric(merged["context_close"], errors="coerce")
    denom = open_close.where(open_close > 0.0)
    merged["episode_runup_from_open_pct"] = (context_close / denom - 1.0).replace([np.inf, -np.inf], np.nan)
    merged["bars_since_episode_open"] = merged["episode_age_bars"]
    episode_state = merged.loc[:, list(EPISODE_STATE_COLUMNS)].copy()
    log_info(
        "LAYERS",
        f"episode_state summary rows_total={len(episode_state)} cols_total={len(episode_state.columns)}",
    )
    stage_done("LAYERS", "EPISODE_STATE", elapsed_sec=time.perf_counter() - started)
    return episode_state
