from __future__ import annotations

import math
import time

import pandas as pd

from pump_end_v2.contracts import OutcomeClass
from pump_end_v2.logging import log_info, stage_done, stage_start


def build_episode_summary(resolved_rows: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("METRICS", "BUILD_EPISODE_SUMMARY")
    if resolved_rows.empty:
        out = pd.DataFrame(
            columns=[
                "episode_id",
                "symbol",
                "episode_open_time",
                "episode_close_time",
                "duration_bars",
                "runup_pct_at_open",
                "total_rows",
                "resolved_rows",
                "good_row_count",
                "good_zone_width_bars",
                "ideal_entry_row_id",
                "ideal_entry_bar_open_time",
                "ideal_pullback_pct",
                "ideal_prepullback_squeeze_pct",
                "ideal_net_edge_pct",
                "bars_open_to_ideal_entry",
                "extension_open_to_ideal_entry_pct",
                "episode_outcome_class",
            ]
        )
        log_info("METRICS", "summary episodes_total=0 good_episode_share=0.0000")
        stage_done("METRICS", "BUILD_EPISODE_SUMMARY", elapsed_sec=time.perf_counter() - started)
        return out
    summary_rows: list[dict[str, object]] = []
    for episode_id, gdf in resolved_rows.groupby("episode_id", sort=False):
        gdf = gdf.sort_values("context_bar_open_time", kind="mergesort").reset_index(drop=True)
        total_rows = len(gdf)
        resolved_count = int(gdf["is_resolved"].sum())
        good_row_count = int((gdf["target_good_short_now"] == 1).sum())
        good_zone_width_bars = int((gdf["target_good_short_now"] == 1).sum())
        ideal_row = gdf[gdf["is_ideal_entry"]]
        if ideal_row.empty:
            ideal_entry_row_id = pd.NA
            ideal_entry_bar_open_time = pd.NaT
            ideal_pullback_pct = math.nan
            ideal_prepullback_squeeze_pct = math.nan
            ideal_net_edge_pct = math.nan
            bars_open_to_ideal_entry = math.nan
            extension_open_to_ideal_entry_pct = math.nan
        else:
            ideal = ideal_row.iloc[0]
            ideal_entry_row_id = ideal["decision_row_id"]
            ideal_entry_bar_open_time = ideal["entry_bar_open_time"]
            ideal_pullback_pct = float(ideal["future_pullback_pct"])
            ideal_prepullback_squeeze_pct = float(ideal["future_prepullback_squeeze_pct"])
            ideal_net_edge_pct = float(ideal["future_net_edge_pct"])
            bars_open_to_ideal_entry = int(
                (ideal_entry_bar_open_time - gdf["episode_open_time"].iloc[0]).total_seconds() // (15 * 60)
            )
            episode_open_close = pd.to_numeric(gdf["episode_open_close"].iloc[0], errors="coerce")
            ideal_entry_price = pd.to_numeric(ideal["entry_price"], errors="coerce")
            if pd.notna(episode_open_close) and float(episode_open_close) > 0.0 and pd.notna(ideal_entry_price):
                extension_open_to_ideal_entry_pct = float(
                    (float(ideal_entry_price) - float(episode_open_close)) / float(episode_open_close)
                )
            else:
                extension_open_to_ideal_entry_pct = math.nan
        runup_pct_at_open = float(pd.to_numeric(gdf["runup_pct_at_context"].iloc[0], errors="coerce"))
        if good_row_count > 0:
            episode_outcome_class = OutcomeClass.REVERSAL.value
        elif (gdf["future_outcome_class"] == OutcomeClass.CONTINUATION.value).any():
            episode_outcome_class = OutcomeClass.CONTINUATION.value
        else:
            episode_outcome_class = OutcomeClass.FLAT.value
        summary_rows.append(
            {
                "episode_id": episode_id,
                "symbol": gdf["symbol"].iloc[0],
                "episode_open_time": gdf["episode_open_time"].iloc[0],
                "episode_close_time": gdf["context_bar_open_time"].max(),
                "duration_bars": int(gdf["episode_age_bars"].max()),
                "runup_pct_at_open": runup_pct_at_open,
                "total_rows": total_rows,
                "resolved_rows": resolved_count,
                "good_row_count": good_row_count,
                "good_zone_width_bars": good_zone_width_bars,
                "ideal_entry_row_id": ideal_entry_row_id,
                "ideal_entry_bar_open_time": ideal_entry_bar_open_time,
                "ideal_pullback_pct": ideal_pullback_pct,
                "ideal_prepullback_squeeze_pct": ideal_prepullback_squeeze_pct,
                "ideal_net_edge_pct": ideal_net_edge_pct,
                "bars_open_to_ideal_entry": bars_open_to_ideal_entry,
                "extension_open_to_ideal_entry_pct": extension_open_to_ideal_entry_pct,
                "episode_outcome_class": episode_outcome_class,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    good_episode_share = (
        float((summary_df["episode_outcome_class"] == OutcomeClass.REVERSAL.value).mean())
        if not summary_df.empty
        else 0.0
    )
    log_info(
        "METRICS",
        f"summary episodes_total={len(summary_df)} good_episode_share={good_episode_share:.4f}",
    )
    stage_done("METRICS", "BUILD_EPISODE_SUMMARY", elapsed_sec=time.perf_counter() - started)
    return summary_df


def build_event_quality_report(summary_df: pd.DataFrame) -> dict[str, float]:
    if summary_df.empty:
        return {
            "episodes_total": 0,
            "episodes_per_30d": 0.0,
            "reversal_share": 0.0,
            "continuation_share": 0.0,
            "flat_share": 0.0,
            "median_bars_open_to_ideal_entry": math.nan,
            "median_ideal_pullback_pct": math.nan,
            "median_ideal_remaining_squeeze_pct": math.nan,
            "median_runup_pct_at_open": math.nan,
            "median_extension_open_to_ideal_entry_pct": math.nan,
            "good_episode_share": 0.0,
            "mean_good_zone_width_bars": 0.0,
        }
    episodes_total = int(len(summary_df))
    open_times = pd.to_datetime(summary_df["episode_open_time"], utc=True, errors="coerce")
    span_days = (open_times.max() - open_times.min()).total_seconds() / 86400.0
    if span_days <= 0:
        episodes_per_30d = float(episodes_total) * 30.0
    else:
        episodes_per_30d = float(episodes_total) / (span_days / 30.0)
    reversal_share = float((summary_df["episode_outcome_class"] == OutcomeClass.REVERSAL.value).mean())
    continuation_share = float((summary_df["episode_outcome_class"] == OutcomeClass.CONTINUATION.value).mean())
    flat_share = float((summary_df["episode_outcome_class"] == OutcomeClass.FLAT.value).mean())
    return {
        "episodes_total": episodes_total,
        "episodes_per_30d": episodes_per_30d,
        "reversal_share": reversal_share,
        "continuation_share": continuation_share,
        "flat_share": flat_share,
        "median_bars_open_to_ideal_entry": float(summary_df["bars_open_to_ideal_entry"].median()),
        "median_ideal_pullback_pct": float(summary_df["ideal_pullback_pct"].median()),
        "median_ideal_remaining_squeeze_pct": float(summary_df["ideal_prepullback_squeeze_pct"].median()),
        "median_runup_pct_at_open": float(summary_df["runup_pct_at_open"].median()),
        "median_extension_open_to_ideal_entry_pct": float(summary_df["extension_open_to_ideal_entry_pct"].median()),
        "good_episode_share": reversal_share,
        "mean_good_zone_width_bars": float(summary_df["good_zone_width_bars"].mean()),
    }
