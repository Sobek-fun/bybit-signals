import pandas as pd

from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.execution.metrics import (
    build_execution_metrics,
    build_execution_window_report,
)
from pump_end_v2.execution.replay import replay_short_signals_with_symbol_lock
from pump_end_v2.logging import log_info

_SWEEP_COLUMNS: tuple[str, ...] = (
    "block_threshold",
    "signals_before",
    "signals_after",
    "blocked_share",
    "bad_block_rate",
    "good_keep_rate",
    "blocked_bad_precision",
    "good_block_tax",
    "signals_per_30d_after",
    "density_penalty",
    "selection_score",
)

_APPLY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "p_block",
)

_EXECUTION_SWEEP_COLUMNS: tuple[str, ...] = (
    "block_threshold",
    "signals_before",
    "signals_after_model",
    "signals_after_execution",
    "blocked_by_model",
    "blocked_by_symbol_lock",
    "failed_execution_other",
    "signals_per_30d_after_execution",
    "pnl_after_execution",
    "worst_6h_after_execution",
    "worst_24h_after_execution",
    "max_losing_streak_after_execution",
    "tp_tax_model",
    "sl_capture_model",
    "tp_tax_execution",
    "sl_capture_execution",
    "density_penalty",
    "selection_score",
)

_COUNTERFACTUAL_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "counterfactual_execution_status",
    "counterfactual_trade_outcome",
    "counterfactual_trade_pnl_pct",
    "counterfactual_exit_time",
)


def build_gate_threshold_grid(base_block_threshold: float) -> list[float]:
    base = float(base_block_threshold)
    raw = [
        base - 0.20,
        base - 0.10,
        base - 0.05,
        base,
        base + 0.05,
        base + 0.10,
        base + 0.20,
    ]
    clipped = [_round6(_clip(value, 0.05, 0.95)) for value in raw]
    return sorted(set(clipped))


def apply_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    block_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(scored_signals_df, _APPLY_REQUIRED_COLUMNS, "scored_signals_df")
    threshold = float(block_threshold)
    decision_df = scored_signals_df.copy()
    decision_df["gate_decision"] = (
        decision_df["p_block"]
        .astype(float)
        .ge(threshold)
        .map({True: "block", False: "keep"})
    )
    decision_df["gate_block_threshold"] = threshold
    kept_signals_df = decision_df[decision_df["gate_decision"] == "keep"].copy()
    return decision_df, kept_signals_df


def build_gate_threshold_metrics(
    decision_df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> dict[str, float]:
    _require_columns(
        decision_df,
        ("target_block_signal", "gate_decision", "context_bar_open_time"),
        "decision_df",
    )
    eval_df = decision_df.copy()
    if "gate_trainable_signal" in eval_df.columns:
        eval_df = eval_df[eval_df["gate_trainable_signal"].astype(bool)].copy()
    signals_before = int(len(eval_df))
    blocked_mask = eval_df["gate_decision"] == "block"
    kept_mask = eval_df["gate_decision"] == "keep"
    bad_mask = (
        pd.to_numeric(eval_df["target_block_signal"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    good_mask = ~bad_mask
    total_blocked = int(blocked_mask.sum())
    total_kept = int(kept_mask.sum())
    total_bad = int(bad_mask.sum())
    total_good = int(good_mask.sum())
    blocked_bad = int((blocked_mask & bad_mask).sum())
    blocked_good = int((blocked_mask & good_mask).sum())
    kept_good = int((kept_mask & good_mask).sum())
    bad_block_rate = _safe_ratio(blocked_bad, total_bad)
    good_keep_rate = _safe_ratio(kept_good, total_good)
    blocked_bad_precision = _safe_ratio(blocked_bad, total_blocked)
    good_block_tax = _safe_ratio(blocked_good, total_good)
    blocked_share = _safe_ratio(total_blocked, signals_before)
    eval_window_days = _resolve_eval_window_days(
        window_start=window_start, window_end=window_end, window_days=window_days
    )
    signals_per_30d_after = _compute_signals_per_30d_after(total_kept, eval_window_days)
    density_penalty = _compute_density_penalty(signals_per_30d_after)
    selection_score = (
        bad_block_rate
        + good_keep_rate
        + blocked_bad_precision
        - good_block_tax
        - 0.25 * density_penalty
    )
    return {
        "signals_before": float(signals_before),
        "signals_after": float(total_kept),
        "blocked_share": float(blocked_share),
        "bad_block_rate": float(bad_block_rate),
        "good_keep_rate": float(good_keep_rate),
        "blocked_bad_precision": float(blocked_bad_precision),
        "good_block_tax": float(good_block_tax),
        "signals_per_30d_after": float(signals_per_30d_after),
        "density_penalty": float(density_penalty),
        "selection_score": float(selection_score),
    }


def sweep_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    base_block_threshold: float,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in build_gate_threshold_grid(base_block_threshold):
        decision_df, _ = apply_gate_block_threshold(scored_signals_df, threshold)
        metrics = build_gate_threshold_metrics(
            decision_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        rows.append({"block_threshold": threshold, **metrics})
    out = pd.DataFrame(rows, columns=list(_SWEEP_COLUMNS))
    log_info("GATE", f"gate threshold sweep done candidates_total={len(out)}")
    return out


def select_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    base_block_threshold: float,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> tuple[float, pd.DataFrame]:
    sweep_df = sweep_gate_block_threshold(
        scored_signals_df,
        base_block_threshold,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
    )
    if sweep_df.empty:
        raise ValueError("gate threshold sweep returned no candidates")
    ranked = sweep_df.sort_values(
        by=[
            "selection_score",
            "blocked_bad_precision",
            "good_keep_rate",
            "density_penalty",
            "block_threshold",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_threshold = float(ranked.iloc[0]["block_threshold"])
    best_selection_score = float(ranked.iloc[0]["selection_score"])
    log_info(
        "GATE",
        f"gate threshold select done best_threshold={best_threshold:.6f} best_selection_score={best_selection_score:.6f}",
    )
    return best_threshold, sweep_df


def build_gate_decile_report(
    scored_signals_df: pd.DataFrame, deciles: int = 10
) -> pd.DataFrame:
    _require_columns(
        scored_signals_df,
        (
            "p_block",
            "future_net_edge_pct",
            "future_pullback_pct",
            "future_prepullback_squeeze_pct",
            "signal_quality_h32",
        ),
        "scored_signals_df",
    )
    frame = scored_signals_df.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "decile",
                "signals",
                "p_block_min",
                "p_block_max",
                "sl_rate",
                "tp_rate",
                "avg_future_net_edge_pct",
                "mean_pullback_pct",
                "mean_squeeze_pct",
            ]
        )
    frame["p_block"] = pd.to_numeric(frame["p_block"], errors="coerce")
    frame["future_net_edge_pct"] = pd.to_numeric(
        frame["future_net_edge_pct"], errors="coerce"
    )
    frame["future_pullback_pct"] = pd.to_numeric(
        frame["future_pullback_pct"], errors="coerce"
    )
    frame["future_prepullback_squeeze_pct"] = pd.to_numeric(
        frame["future_prepullback_squeeze_pct"], errors="coerce"
    )
    frame = frame.dropna(subset=["p_block"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["decile", "signals"])
    bins = min(int(deciles), int(frame["p_block"].nunique()))
    if bins <= 1:
        frame["decile"] = 1
    else:
        frame["decile"] = (
            pd.qcut(frame["p_block"], q=bins, labels=False, duplicates="drop") + 1
        )
    rows: list[dict[str, float]] = []
    for decile, gdf in frame.groupby("decile", sort=True):
        quality = gdf["signal_quality_h32"].astype(str)
        tp_rate = (
            float((quality == "clean_retrace_h32").mean()) if len(gdf) > 0 else 0.0
        )
        sl_rate = (
            float(
                quality.isin(
                    {
                        "dirty_retrace_h32",
                        "clean_no_pullback_h32",
                        "dirty_no_pullback_h32",
                        "pullback_before_squeeze_h32",
                    }
                ).mean()
            )
            if len(gdf) > 0
            else 0.0
        )
        rows.append(
            {
                "decile": float(decile),
                "signals": float(len(gdf)),
                "p_block_min": float(gdf["p_block"].min()),
                "p_block_max": float(gdf["p_block"].max()),
                "sl_rate": sl_rate,
                "tp_rate": tp_rate,
                "avg_future_net_edge_pct": (
                    float(gdf["future_net_edge_pct"].mean() * 100.0)
                    if len(gdf) > 0
                    else float("nan")
                ),
                "mean_pullback_pct": (
                    float(gdf["future_pullback_pct"].mean() * 100.0)
                    if len(gdf) > 0
                    else float("nan")
                ),
                "mean_squeeze_pct": (
                    float(gdf["future_prepullback_squeeze_pct"].mean() * 100.0)
                    if len(gdf) > 0
                    else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("decile", kind="mergesort")
        .reset_index(drop=True)
    )


def select_gate_block_threshold_execution_aware(
    scored_signals_df: pd.DataFrame,
    base_block_threshold: float,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_df: pd.DataFrame | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> tuple[float, pd.DataFrame]:
    counterfactual_outcomes_df = attach_counterfactual_execution_outcomes(
        scored_signals_df,
        bars_1m_df,
        execution_contract,
        bars_1s_df,
    )
    rows: list[dict[str, float]] = []
    for threshold in build_gate_threshold_grid(base_block_threshold):
        gate_decisions_df, _ = apply_gate_block_threshold(scored_signals_df, threshold)
        execution_decisions_df, executed_signals_df = (
            replay_short_signals_with_symbol_lock(
                gate_decisions_df,
                bars_1m_df,
                execution_contract,
                bars_1s_df,
            )
        )
        execution_decisions_df = execution_decisions_df.merge(
            counterfactual_outcomes_df,
            on="signal_id",
            how="left",
            validate="one_to_one",
        )
        execution_metrics = build_execution_metrics(
            executed_signals_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        window_6h = build_execution_window_report(executed_signals_df, 6)
        window_24h = build_execution_window_report(executed_signals_df, 24)
        summary = build_gate_execution_decision_summary(execution_decisions_df)
        signals_per_30d = float(execution_metrics.get("signals_per_30d", 0.0))
        pnl_sum = float(execution_metrics.get("pnl_sum", 0.0))
        worst_6h = float(window_6h["pnl_sum"].min()) if not window_6h.empty else 0.0
        worst_24h = float(window_24h["pnl_sum"].min()) if not window_24h.empty else 0.0
        max_losing_streak = float(execution_metrics.get("max_losing_streak", 0.0))
        density_penalty = _compute_density_penalty(signals_per_30d)
        selection_score = (
            pnl_sum
            + 5.0 * float(summary["sl_capture_execution"])
            - 5.0 * float(summary["tp_tax_execution"])
            - 0.75 * density_penalty
            + 0.25 * worst_6h
            + 0.10 * worst_24h
            - 0.10 * max_losing_streak
        )
        rows.append(
            {
                "block_threshold": float(threshold),
                "signals_before": float(summary["candidate_signals_before"]),
                "signals_after_model": float(summary["after_model"]),
                "signals_after_execution": float(summary["after_execution"]),
                "blocked_by_model": float(summary["blocked_by_model"]),
                "blocked_by_symbol_lock": float(summary["blocked_by_symbol_lock"]),
                "failed_execution_other": float(summary["failed_execution_other"]),
                "signals_per_30d_after_execution": signals_per_30d,
                "pnl_after_execution": pnl_sum,
                "worst_6h_after_execution": worst_6h,
                "worst_24h_after_execution": worst_24h,
                "max_losing_streak_after_execution": max_losing_streak,
                "tp_tax_model": float(summary["tp_tax_model"]),
                "sl_capture_model": float(summary["sl_capture_model"]),
                "tp_tax_execution": float(summary["tp_tax_execution"]),
                "sl_capture_execution": float(summary["sl_capture_execution"]),
                "density_penalty": float(density_penalty),
                "selection_score": float(selection_score),
            }
        )
    sweep_df = pd.DataFrame(rows, columns=list(_EXECUTION_SWEEP_COLUMNS))
    if sweep_df.empty:
        raise ValueError("execution-aware gate threshold sweep returned no candidates")
    ranked = sweep_df.sort_values(
        by=[
            "selection_score",
            "pnl_after_execution",
            "signals_per_30d_after_execution",
            "tp_tax_execution",
            "block_threshold",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_threshold = float(ranked.iloc[0]["block_threshold"])
    log_info(
        "GATE",
        (
            "gate threshold execution-aware select done "
            f"best_threshold={best_threshold:.6f} "
            f"best_selection_score={float(ranked.iloc[0]['selection_score']):.6f}"
        ),
    )
    return best_threshold, sweep_df


def build_gate_execution_decision_summary(
    execution_decisions_df: pd.DataFrame,
) -> dict[str, float]:
    _require_columns(
        execution_decisions_df,
        (
            "gate_decision",
            "execution_status",
            "trade_pnl_pct",
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
        ),
        "execution_decisions_df",
    )
    joined = execution_decisions_df.copy()
    gate_keep_mask = joined["gate_decision"].astype(str) == "keep"
    executed_mask = joined["execution_status"].astype(str) == "executed"
    blocked_gate_mask = joined["execution_status"].astype(str) == "blocked_gate"
    blocked_symbol_lock_mask = (
        joined["execution_status"].astype(str) == "blocked_symbol_lock"
    )
    failed_execution_other_mask = gate_keep_mask & ~joined["execution_status"].astype(
        str
    ).isin({"executed", "blocked_symbol_lock"})
    counterfactual_outcome = joined["counterfactual_trade_outcome"].astype(str)
    tp_mask = counterfactual_outcome == "tp"
    sl_mask = counterfactual_outcome == "sl"
    tp_total_all_candidates = int(tp_mask.sum())
    sl_total_all_candidates = int(sl_mask.sum())
    tp_total_kept = int((gate_keep_mask & tp_mask).sum())
    sl_total_kept = int((gate_keep_mask & sl_mask).sum())
    tp_blocked_model = int((blocked_gate_mask & tp_mask).sum())
    sl_blocked_model = int((blocked_gate_mask & sl_mask).sum())
    tp_blocked_execution = int((blocked_symbol_lock_mask & tp_mask).sum())
    sl_blocked_execution = int((blocked_symbol_lock_mask & sl_mask).sum())
    summary = {
        "candidate_signals_before": float(len(joined)),
        "after_model": float(int(gate_keep_mask.sum())),
        "after_execution": float(int(executed_mask.sum())),
        "blocked_by_model": float(int(blocked_gate_mask.sum())),
        "blocked_by_symbol_lock": float(int(blocked_symbol_lock_mask.sum())),
        "failed_execution_other": float(int(failed_execution_other_mask.sum())),
        "tp_blocked_model": float(tp_blocked_model),
        "sl_blocked_model": float(sl_blocked_model),
        "tp_blocked_execution": float(tp_blocked_execution),
        "sl_blocked_execution": float(sl_blocked_execution),
        "tp_tax_model": float(_safe_ratio(tp_blocked_model, tp_total_all_candidates)),
        "sl_capture_model": float(
            _safe_ratio(sl_blocked_model, sl_total_all_candidates)
        ),
        "tp_tax_execution": float(_safe_ratio(tp_blocked_execution, tp_total_kept)),
        "sl_capture_execution": float(_safe_ratio(sl_blocked_execution, sl_total_kept)),
        "pnl_before": float(
            pd.to_numeric(joined["counterfactual_trade_pnl_pct"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "pnl_after_execution": float(
            pd.to_numeric(joined.loc[executed_mask, "trade_pnl_pct"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
    }
    return summary


def _compute_signals_per_30d_after(
    kept_signals_count: int, eval_window_days: float
) -> float:
    if kept_signals_count <= 0:
        return 0.0
    safe_window_days = max(float(eval_window_days), 1e-9)
    return float(float(kept_signals_count) * 30.0 / safe_window_days)


def _compute_density_penalty(signals_per_30d_after: float) -> float:
    x = float(signals_per_30d_after)
    if 30.0 <= x <= 60.0:
        return 0.0
    if x < 30.0:
        return float((30.0 - x) / 30.0)
    return float((x - 60.0) / 60.0)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _round6(value: float) -> float:
    return float(round(float(value), 6))


def _resolve_eval_window_days(
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
    window_days: float | None,
) -> float:
    if window_days is not None:
        resolved = float(window_days)
        if resolved <= 0.0:
            raise ValueError("window_days must be positive")
        return resolved
    if window_start is None or window_end is None:
        return 1.0
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    if end <= start:
        raise ValueError("window_end must be greater than window_start")
    return float((end - start) / pd.Timedelta(days=1))


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def attach_counterfactual_execution_outcomes(
    decision_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_columns(
        decision_df,
        (
            "signal_id",
            "episode_id",
            "symbol",
            "context_bar_open_time",
            "decision_time",
            "entry_bar_open_time",
        ),
        "decision_df",
    )
    if decision_df.empty:
        return pd.DataFrame(columns=list(_COUNTERFACTUAL_COLUMNS))
    identity_columns = [
        "signal_id",
        "episode_id",
        "symbol",
        "context_bar_open_time",
        "decision_time",
        "entry_bar_open_time",
    ]
    rows: list[dict[str, object]] = []
    for row in decision_df.loc[:, identity_columns].itertuples(index=False):
        one_signal = pd.DataFrame(
            [
                {
                    "signal_id": row.signal_id,
                    "episode_id": row.episode_id,
                    "symbol": row.symbol,
                    "context_bar_open_time": row.context_bar_open_time,
                    "decision_time": row.decision_time,
                    "entry_bar_open_time": row.entry_bar_open_time,
                    "gate_decision": "keep",
                }
            ]
        )
        one_decision_df, _ = replay_short_signals_with_symbol_lock(
            one_signal,
            bars_1m_df,
            execution_contract,
            bars_1s_df,
        )
        one = one_decision_df.iloc[0]
        rows.append(
            {
                "signal_id": row.signal_id,
                "counterfactual_execution_status": one.get("execution_status", pd.NA),
                "counterfactual_trade_outcome": one.get("trade_outcome", pd.NA),
                "counterfactual_trade_pnl_pct": one.get("trade_pnl_pct", pd.NA),
                "counterfactual_exit_time": one.get("exit_time", pd.NaT),
            }
        )
    out = pd.DataFrame(rows, columns=list(_COUNTERFACTUAL_COLUMNS))
    if not decision_df["signal_id"].is_unique:
        raise ValueError(
            "decision_df must have unique signal_id for counterfactual enrichment"
        )
    if not out["signal_id"].is_unique:
        raise ValueError("counterfactual outcomes produced duplicate signal_id")
    return out
