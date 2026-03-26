import math
import time

import pandas as pd

from pump_end_v2.config import ResolverConfig
from pump_end_v2.contracts import OutcomeClass, SignalQualityClass, TargetReason
from pump_end_v2.logging import log_info, stage_done, stage_start


def resolve_decision_rows(
    df: pd.DataFrame, decision_rows: pd.DataFrame, config: ResolverConfig
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("RESOLVER", "RESOLVE_ROWS")
    if decision_rows.empty:
        out = decision_rows.copy()
        log_info(
            "RESOLVER",
            "summary rows_total=0 resolved_rows=0 good_rows=0 reversal_share=0.0000",
        )
        stage_done(
            "RESOLVER", "RESOLVE_ROWS", elapsed_sec=time.perf_counter() - started
        )
        return out
    resolved = decision_rows.copy()
    resolved["is_resolved"] = False
    resolved["entry_price"] = math.nan
    resolved["future_prepullback_squeeze_pct"] = math.nan
    resolved["future_pullback_pct"] = math.nan
    resolved["future_net_edge_pct"] = math.nan
    resolved["bars_to_pullback"] = pd.NA
    resolved["bars_to_peak_after_row"] = pd.NA
    resolved["bars_to_resolution"] = pd.NA
    resolved["future_outcome_class"] = pd.NA
    resolved["signal_quality_h32"] = pd.NA
    resolved["target_good_short_now"] = 0
    resolved["target_reason"] = TargetReason.INVALID_CONTEXT.value
    resolved["entry_quality_score"] = math.nan
    resolved["ideal_entry_row_id"] = pd.NA
    resolved["ideal_entry_bar_open_time"] = pd.Series(
        pd.NaT,
        index=resolved.index,
        dtype=resolved["entry_bar_open_time"].dtype,
    )
    resolved["is_ideal_entry"] = False
    market_by_symbol = {
        symbol: sdf.reset_index(drop=True).copy()
        for symbol, sdf in df.groupby("symbol", sort=False)
    }
    for idx, row in resolved.iterrows():
        symbol_df = market_by_symbol.get(row["symbol"])
        if symbol_df is None:
            continue
        entry_mask = symbol_df["open_time"] == row["entry_bar_open_time"]
        if not entry_mask.any():
            continue
        entry_pos = int(entry_mask[entry_mask].index[0])
        horizon_df = symbol_df.iloc[entry_pos : entry_pos + config.horizon_bars].copy()
        if len(horizon_df) < config.horizon_bars:
            continue
        entry_price = float(horizon_df["open"].iloc[0])
        if entry_price <= 0:
            continue
        squeeze_series = (horizon_df["high"] - entry_price) / entry_price
        pullback_series = (entry_price - horizon_df["low"]) / entry_price
        max_squeeze_pct_total = float(squeeze_series.max())
        max_pullback_pct_total = float(pullback_series.max())
        success_mask = horizon_df["low"] <= entry_price * (
            1.0 - config.success_pullback_pct
        )
        first_success_pullback_index = (
            int(success_mask[success_mask].index[0] - horizon_df.index[0])
            if success_mask.any()
            else None
        )
        if first_success_pullback_index is None:
            future_prepullback_squeeze_pct = max_squeeze_pct_total
        else:
            future_prepullback_squeeze_pct = float(
                squeeze_series.iloc[: first_success_pullback_index + 1].max()
            )
        peak_high = float(horizon_df["high"].max())
        first_peak_index = int(
            (horizon_df["high"] == peak_high).idxmax() - horizon_df.index[0]
        )
        bars_to_resolution = (
            first_success_pullback_index
            if first_success_pullback_index is not None
            else int(config.horizon_bars)
        )
        future_pullback_pct = max_pullback_pct_total
        future_net_edge_pct = future_pullback_pct - future_prepullback_squeeze_pct
        first_squeeze_breach_index = _first_true_index(
            squeeze_series > float(config.max_prepullback_squeeze_pct),
            int(horizon_df.index[0]),
        )
        has_success_pullback = first_success_pullback_index is not None
        wait_ok = has_success_pullback and int(first_success_pullback_index) <= int(
            config.max_wait_bars_for_success
        )
        squeeze_ok = future_prepullback_squeeze_pct <= float(
            config.max_prepullback_squeeze_pct
        )
        pullback_before_squeeze = (
            has_success_pullback
            and first_squeeze_breach_index is not None
            and int(first_success_pullback_index) < int(first_squeeze_breach_index)
        )
        is_reversal = has_success_pullback and squeeze_ok and wait_ok
        if is_reversal:
            signal_quality = SignalQualityClass.CLEAN_RETRACE_H32.value
        elif has_success_pullback and pullback_before_squeeze:
            signal_quality = SignalQualityClass.PULLBACK_BEFORE_SQUEEZE_H32.value
        elif has_success_pullback:
            signal_quality = SignalQualityClass.DIRTY_RETRACE_H32.value
        elif max_squeeze_pct_total <= config.flat_max_abs_move_pct:
            signal_quality = SignalQualityClass.CLEAN_NO_PULLBACK_H32.value
        else:
            signal_quality = SignalQualityClass.DIRTY_NO_PULLBACK_H32.value
        if is_reversal:
            outcome = OutcomeClass.REVERSAL.value
        elif max_squeeze_pct_total > config.flat_max_abs_move_pct:
            outcome = OutcomeClass.CONTINUATION.value
        else:
            outcome = OutcomeClass.FLAT.value
        resolved.at[idx, "is_resolved"] = True
        resolved.at[idx, "entry_price"] = entry_price
        resolved.at[idx, "future_prepullback_squeeze_pct"] = (
            future_prepullback_squeeze_pct
        )
        resolved.at[idx, "future_pullback_pct"] = future_pullback_pct
        resolved.at[idx, "future_net_edge_pct"] = future_net_edge_pct
        resolved.at[idx, "bars_to_pullback"] = first_success_pullback_index
        resolved.at[idx, "bars_to_peak_after_row"] = first_peak_index
        resolved.at[idx, "bars_to_resolution"] = bars_to_resolution
        resolved.at[idx, "future_outcome_class"] = outcome
        resolved.at[idx, "signal_quality_h32"] = signal_quality
        resolved.at[idx, "target_good_short_now"] = (
            1 if outcome == OutcomeClass.REVERSAL.value else 0
        )
        resolved.at[idx, "entry_quality_score"] = future_net_edge_pct - 0.001 * min(
            bars_to_resolution, config.horizon_bars
        )
    resolved = _attach_episode_ideal_entry(resolved)
    resolved = _apply_target_reason(resolved)
    resolved_rows_count = int(resolved["is_resolved"].sum())
    good_rows = int((resolved["target_good_short_now"] == 1).sum())
    reversal_share = good_rows / resolved_rows_count if resolved_rows_count else 0.0
    log_info(
        "RESOLVER",
        (
            f"summary rows_total={len(resolved)} "
            f"resolved_rows={resolved_rows_count} good_rows={good_rows} reversal_share={reversal_share:.4f}"
        ),
    )
    stage_done("RESOLVER", "RESOLVE_ROWS", elapsed_sec=time.perf_counter() - started)
    return resolved


def _attach_episode_ideal_entry(resolved: pd.DataFrame) -> pd.DataFrame:
    frame = resolved.copy()
    for episode_id, gdf in frame.groupby("episode_id", sort=False):
        candidates = gdf[gdf["is_resolved"]].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            by=["entry_quality_score", "entry_bar_open_time", "decision_row_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        ideal = candidates.iloc[0]
        episode_mask = frame["episode_id"] == episode_id
        frame.loc[episode_mask, "ideal_entry_row_id"] = ideal["decision_row_id"]
        frame.loc[episode_mask, "ideal_entry_bar_open_time"] = ideal[
            "entry_bar_open_time"
        ]
        frame.loc[episode_mask, "is_ideal_entry"] = (
            frame.loc[episode_mask, "decision_row_id"] == ideal["decision_row_id"]
        )
    return frame


def _apply_target_reason(resolved: pd.DataFrame) -> pd.DataFrame:
    frame = resolved.copy()
    good_by_episode = (
        frame.groupby("episode_id")["target_good_short_now"]
        .max()
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    for idx, row in frame.iterrows():
        if not bool(row["is_resolved"]):
            frame.at[idx, "target_reason"] = TargetReason.INVALID_CONTEXT.value
            continue
        if int(row["target_good_short_now"]) == 1:
            frame.at[idx, "target_reason"] = TargetReason.GOOD.value
            continue
        if good_by_episode.get(row["episode_id"], 0) > 0 and pd.notna(
            row["ideal_entry_bar_open_time"]
        ):
            if row["entry_bar_open_time"] < row["ideal_entry_bar_open_time"]:
                frame.at[idx, "target_reason"] = TargetReason.TOO_EARLY.value
                continue
            if row["entry_bar_open_time"] > row["ideal_entry_bar_open_time"]:
                frame.at[idx, "target_reason"] = TargetReason.TOO_LATE.value
                continue
        if row["future_outcome_class"] == OutcomeClass.CONTINUATION.value:
            frame.at[idx, "target_reason"] = TargetReason.CONTINUATION.value
        elif row["future_outcome_class"] == OutcomeClass.FLAT.value:
            frame.at[idx, "target_reason"] = TargetReason.FLAT.value
        else:
            frame.at[idx, "target_reason"] = TargetReason.INVALID_CONTEXT.value
    return frame


def _first_true_index(mask: pd.Series, base_index: int) -> int | None:
    true_mask = mask[mask]
    if true_mask.empty:
        return None
    return int(true_mask.index[0] - base_index)
