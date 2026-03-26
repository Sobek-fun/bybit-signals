from __future__ import annotations

import time

import pandas as pd

from pump_end_v2.contracts import DecisionRowRef, ExecutionContract
from pump_end_v2.logging import log_info, stage_done, stage_start
from pump_end_v2.time_utils import context_to_decision_time, decision_to_entry_bar_open_time

DECISION_ROW_COLUMNS: tuple[str, ...] = (
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "episode_age_bars",
    "episode_open_time",
    "episode_open_close",
    "episode_high_so_far",
    "distance_from_episode_high_pct",
    "runup_pct_at_context",
    "volume_ratio_at_context",
    "pump_context_flag",
)


def build_decision_rows(
    token_state_df: pd.DataFrame, episodes: pd.DataFrame, execution: ExecutionContract
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("ROWS", "BUILD_DECISION_ROWS")
    if episodes.empty:
        out = pd.DataFrame(columns=list(DECISION_ROW_COLUMNS))
        log_info("ROWS", "summary episodes_total=0 rows_total=0")
        stage_done("ROWS", "BUILD_DECISION_ROWS", elapsed_sec=time.perf_counter() - started)
        return out
    context_by_symbol = {
        symbol: sdf.reset_index(drop=True).copy() for symbol, sdf in token_state_df.groupby("symbol", sort=False)
    }
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        symbol_context = context_by_symbol.get(episode.symbol)
        if symbol_context is None:
            continue
        episode_slice = symbol_context[
            (symbol_context["open_time"] >= episode.episode_open_time)
            & (symbol_context["open_time"] <= episode.episode_close_time)
        ].copy()
        if episode_slice.empty:
            continue
        episode_slice["episode_age_bars"] = range(1, len(episode_slice) + 1)
        episode_slice["episode_high_so_far"] = episode_slice["high"].cummax()
        episode_slice["distance_from_episode_high_pct"] = (
            (episode_slice["episode_high_so_far"] - episode_slice["close"]) / episode_slice["episode_high_so_far"]
        )
        for row in episode_slice.itertuples(index=False):
            decision_time = context_to_decision_time(row.open_time)
            entry_bar_open_time = decision_to_entry_bar_open_time(
                decision_time, entry_shift_bars=execution.entry_shift_bars
            )
            decision_row_id = f"{episode.episode_id}|{row.open_time:%Y%m%d_%H%M%S}"
            _ = DecisionRowRef(
                decision_row_id=decision_row_id,
                episode_id=str(episode.episode_id),
                symbol=str(episode.symbol),
                context_bar_open_time=pd.Timestamp(row.open_time).to_pydatetime(),
                decision_time=pd.Timestamp(decision_time).to_pydatetime(),
                entry_bar_open_time=pd.Timestamp(entry_bar_open_time).to_pydatetime(),
            )
            rows.append(
                {
                    "decision_row_id": decision_row_id,
                    "episode_id": episode.episode_id,
                    "symbol": episode.symbol,
                    "context_bar_open_time": row.open_time,
                    "decision_time": decision_time,
                    "entry_bar_open_time": entry_bar_open_time,
                    "episode_age_bars": int(row.episode_age_bars),
                    "episode_open_time": episode.episode_open_time,
                    "episode_open_close": float(episode.episode_open_close),
                    "episode_high_so_far": float(row.episode_high_so_far),
                    "distance_from_episode_high_pct": float(row.distance_from_episode_high_pct),
                    "runup_pct_at_context": float(row.runup_pct),
                    "volume_ratio_at_context": float(row.volume_ratio),
                    "pump_context_flag": bool(row.pump_context_flag),
                }
            )
    decision_rows = pd.DataFrame(rows)
    if decision_rows.empty:
        decision_rows = pd.DataFrame(columns=list(DECISION_ROW_COLUMNS))
    else:
        decision_rows = decision_rows.loc[:, list(DECISION_ROW_COLUMNS)].copy()
    log_info(
        "ROWS",
        f"summary episodes_total={len(episodes)} rows_total={len(decision_rows)}",
    )
    stage_done("ROWS", "BUILD_DECISION_ROWS", elapsed_sec=time.perf_counter() - started)
    return decision_rows
