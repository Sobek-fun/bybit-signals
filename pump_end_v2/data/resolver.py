import time

import pandas as pd

from pump_end_v2.config import ResolverConfig
from pump_end_v2.contracts import ExecutionContract, TargetReason, TradeOutcome
from pump_end_v2.execution.replay import (
    ExecutionMarketView,
    OneSecondBarsFetcher,
    replay_independent_short_signals,
)
from pump_end_v2.logging import log_info, stage_done, stage_start


def resolve_decision_rows(
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    decision_rows: pd.DataFrame,
    execution_contract: ExecutionContract,
    config: ResolverConfig,
    bars_1s_fetcher: OneSecondBarsFetcher | None = None,
    execution_market_view: ExecutionMarketView | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("RESOLVER", "RESOLVE_ROWS")
    if decision_rows.empty:
        out = decision_rows.copy()
        log_info(
            "RESOLVER",
            "summary rows_total=0 resolved_rows=0 good_rows=0 tp_share=0.0000",
        )
        stage_done(
            "RESOLVER", "RESOLVE_ROWS", elapsed_sec=time.perf_counter() - started
        )
        return out
    resolved = decision_rows.copy()
    for col in ("context_bar_open_time", "decision_time", "entry_bar_open_time"):
        resolved[col] = pd.to_datetime(resolved[col], utc=True, errors="raise")
    decision_df = resolved.loc[
        :,
        [
            "decision_row_id",
            "episode_id",
            "symbol",
            "context_bar_open_time",
            "decision_time",
            "entry_bar_open_time",
        ],
    ].copy()
    decision_df = decision_df.rename(columns={"decision_row_id": "signal_id"})
    outcomes_df = replay_independent_short_signals(
        decision_df=decision_df,
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
        execution_contract=execution_contract,
        bars_1s_fetcher=bars_1s_fetcher,
        market_view=execution_market_view,
    )
    merged = resolved.merge(
        outcomes_df.rename(
            columns={
                "signal_id": "decision_row_id",
                "trade_outcome": "row_trade_outcome",
                "trade_pnl_pct": "row_trade_pnl_pct",
                "mfe_pct": "row_mfe_pct",
                "mae_pct": "row_mae_pct",
                "holding_bars": "row_holding_bars",
            }
        ),
        on="decision_row_id",
        how="left",
        validate="one_to_one",
    )
    merged["is_resolved"] = merged["execution_status"].astype(str).eq("executed")
    merged["target_good_short_now"] = 0
    merged["target_reason"] = TargetReason.INVALID_CONTEXT.value
    merged["target_row_weight"] = pd.NA
    merged["entry_quality_score"] = pd.to_numeric(
        merged["row_trade_pnl_pct"], errors="coerce"
    )
    merged["future_prepullback_squeeze_pct"] = pd.NA
    merged["future_pullback_pct"] = pd.NA
    merged["future_net_edge_pct"] = pd.NA
    merged["bars_to_pullback"] = pd.NA
    merged["bars_to_peak_after_row"] = pd.NA
    merged["bars_to_resolution"] = pd.NA
    merged["future_outcome_class"] = pd.NA
    merged["signal_quality_h32"] = pd.NA
    merged["ideal_entry_row_id"] = pd.NA
    merged["ideal_entry_bar_open_time"] = pd.Series(
        pd.NaT, index=merged.index, dtype=merged["entry_bar_open_time"].dtype
    )
    merged["is_ideal_entry"] = False
    row_outcome = merged["row_trade_outcome"].astype(str).str.lower()
    tp_mask = merged["is_resolved"] & row_outcome.eq(TradeOutcome.TP.value)
    sl_mask = merged["is_resolved"] & row_outcome.eq(TradeOutcome.SL.value)
    timeout_mask = merged["is_resolved"] & row_outcome.eq(TradeOutcome.TIMEOUT.value)
    ambiguous_mask = merged["is_resolved"] & row_outcome.eq(TradeOutcome.AMBIGUOUS.value)
    merged.loc[tp_mask, "target_good_short_now"] = 1
    merged.loc[tp_mask, "target_reason"] = TargetReason.TP.value
    merged.loc[tp_mask, "target_row_weight"] = float(config.tp_row_weight)
    merged.loc[sl_mask, "target_reason"] = TargetReason.SL.value
    merged.loc[sl_mask, "target_row_weight"] = float(config.sl_row_weight)
    merged.loc[timeout_mask, "target_reason"] = TargetReason.TIMEOUT.value
    merged.loc[timeout_mask, "target_row_weight"] = float(config.timeout_row_weight)
    merged.loc[ambiguous_mask, "target_reason"] = TargetReason.AMBIGUOUS.value
    merged.loc[ambiguous_mask, "target_row_weight"] = pd.NA
    merged.loc[~merged["is_resolved"], "target_reason"] = TargetReason.INVALID_CONTEXT.value
    merged.loc[~merged["is_resolved"], "target_row_weight"] = pd.NA
    resolved = _attach_episode_ideal_entry(merged)
    resolved_rows_count = int(resolved["is_resolved"].sum())
    good_rows = int((resolved["target_good_short_now"] == 1).sum())
    tp_share = good_rows / resolved_rows_count if resolved_rows_count else 0.0
    log_info(
        "RESOLVER",
        (
            f"summary rows_total={len(resolved)} "
            f"resolved_rows={resolved_rows_count} good_rows={good_rows} tp_share={tp_share:.4f}"
        ),
    )
    stage_done("RESOLVER", "RESOLVE_ROWS", elapsed_sec=time.perf_counter() - started)
    return resolved


def _attach_episode_ideal_entry(resolved: pd.DataFrame) -> pd.DataFrame:
    frame = resolved.copy()
    for episode_id, gdf in frame.groupby("episode_id", sort=False):
        candidates = gdf[gdf["target_good_short_now"] == 1].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            by=["entry_bar_open_time", "decision_row_id"],
            ascending=[True, True],
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
