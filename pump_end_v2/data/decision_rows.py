from __future__ import annotations

import pandas as pd

from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.logging import stage_done, stage_start
from pump_end_v2.time_utils import context_to_decision_time, decision_to_entry_bar_open_time

DEFAULT_RUNUP_LOOKBACK_BARS = 12
DEFAULT_MIN_RUNUP_PCT = 0.08
DEFAULT_NEAR_HIGH_LOOKBACK_BARS = 8
DEFAULT_NEAR_HIGH_TOL_PCT = 0.01
DEFAULT_VOLUME_RATIO_LOOKBACK_BARS = 20
DEFAULT_MIN_VOLUME_RATIO = 1.5


def build_decision_rows(df: pd.DataFrame, episodes: pd.DataFrame, execution: ExecutionContract) -> pd.DataFrame:
    stage_start("ROWS", "ROWS")
    if episodes.empty:
        stage_done("ROWS", "ROWS episodes_total=0 rows_total=0")
        return pd.DataFrame(
            columns=[
                "decision_row_id",
                "episode_id",
                "symbol",
                "context_bar_open_time",
                "decision_time",
                "entry_bar_open_time",
                "episode_age_bars",
                "episode_open_time",
                "episode_close_time",
                "duration_bars",
                "episode_high_so_far",
                "distance_from_episode_high_pct",
                "runup_pct_at_context",
                "volume_ratio_at_context",
                "pump_context_flag",
            ]
        )
    context_by_symbol = {symbol: _build_symbol_context(sdf) for symbol, sdf in df.groupby("symbol", sort=False)}
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
            rows.append(
                {
                    "decision_row_id": f"{episode.episode_id}|{row.open_time:%Y%m%d_%H%M%S}",
                    "episode_id": episode.episode_id,
                    "symbol": episode.symbol,
                    "context_bar_open_time": row.open_time,
                    "decision_time": decision_time,
                    "entry_bar_open_time": entry_bar_open_time,
                    "episode_age_bars": int(row.episode_age_bars),
                    "episode_open_time": episode.episode_open_time,
                    "episode_close_time": episode.episode_close_time,
                    "duration_bars": int(episode.duration_bars),
                    "episode_high_so_far": float(row.episode_high_so_far),
                    "distance_from_episode_high_pct": float(row.distance_from_episode_high_pct),
                    "runup_pct_at_context": float(row.runup_pct),
                    "volume_ratio_at_context": float(row.volume_ratio),
                    "pump_context_flag": bool(row.pump_context_flag),
                }
            )
    decision_rows = pd.DataFrame(rows)
    stage_done(
        "ROWS",
        f"ROWS episodes_total={len(episodes)} rows_total={len(decision_rows)}",
    )
    return decision_rows


def _build_symbol_context(sdf: pd.DataFrame) -> pd.DataFrame:
    frame = sdf.reset_index(drop=True).copy()
    frame["rolling_min_low"] = frame["low"].rolling(window=DEFAULT_RUNUP_LOOKBACK_BARS, min_periods=1).min()
    frame["runup_pct"] = frame["close"] / frame["rolling_min_low"] - 1.0
    frame["rolling_max_high"] = frame["high"].rolling(window=DEFAULT_NEAR_HIGH_LOOKBACK_BARS, min_periods=1).max()
    frame["near_high_flag"] = frame["close"] >= frame["rolling_max_high"] * (1.0 - DEFAULT_NEAR_HIGH_TOL_PCT)
    frame["rolling_mean_volume"] = frame["volume"].rolling(window=DEFAULT_VOLUME_RATIO_LOOKBACK_BARS, min_periods=1).mean()
    frame["volume_ratio"] = frame["volume"] / frame["rolling_mean_volume"]
    frame["volume_ratio"] = frame["volume_ratio"].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    frame["pump_context_flag"] = (
        (frame["runup_pct"] >= DEFAULT_MIN_RUNUP_PCT)
        & frame["near_high_flag"]
        & (frame["volume_ratio"] >= DEFAULT_MIN_VOLUME_RATIO)
    )
    return frame
