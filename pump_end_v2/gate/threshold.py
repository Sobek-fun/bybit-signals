import pandas as pd
import sys
import time

from pump_end_v2.config import GateThresholdSearchConfig
from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.execution.metrics import (
    build_execution_metrics,
    build_execution_window_report,
)
from pump_end_v2.execution.replay import (
    replay_independent_short_signals,
    replay_short_signals_with_symbol_lock_precomputed,
)
from pump_end_v2.logging import log_info

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

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
    "support_ok",
    "useful_ok",
    "goal_zone_ok",
    "model_zone_distance",
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
    "gate_mode",
    "block_threshold",
    "signals_before",
    "signals_after_model",
    "signals_after_execution",
    "blocked_by_model",
    "blocked_by_model_trainable",
    "blocked_by_symbol_lock",
    "failed_execution_other",
    "signals_per_30d_after_execution",
    "pnl_after_execution",
    "worst_6h_after_execution",
    "worst_24h_after_execution",
    "max_losing_streak_after_execution",
    "tp_tax_model",
    "sl_capture_model",
    "blocked_sl_precision_model",
    "tp_blocked_model",
    "sl_blocked_model",
    "model_useful",
    "model_zone_distance",
    "tp_tax_execution",
    "sl_capture_execution",
    "model_frontier_admissible",
    "model_frontier_dominated",
    "model_frontier_rank",
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


def build_gate_threshold_grid(
    base_block_threshold: float,
    search_config: GateThresholdSearchConfig | None = None,
    scored_signals_df: pd.DataFrame | None = None,
) -> list[float]:
    if search_config is not None and len(search_config.threshold_candidates) > 0:
        explicit = [
            _round6(_clip(float(value), 0.0, 1.0))
            for value in search_config.threshold_candidates
        ]
        return sorted(set(explicit))
    candidates: set[float] = set()
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
    candidates.update(clipped)
    candidates.update(_build_empirical_threshold_grid(scored_signals_df))
    return sorted(candidates)


def _tqdm_kwargs() -> dict[str, object]:
    return {
        "disable": not sys.stderr.isatty(),
        "leave": False,
        "dynamic_ncols": True,
        "mininterval": 0.5,
    }


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
    search_config: GateThresholdSearchConfig | None = None,
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
    target_sl_capture_model = (
        float(search_config.target_sl_capture_model)
        if search_config is not None
        else 0.20
    )
    max_tp_tax_model = (
        float(search_config.max_tp_tax_model) if search_config is not None else 0.10
    )
    min_blocked_trainable = (
        int(search_config.min_blocked_trainable) if search_config is not None else 10
    )
    require_sl_gt_tp = (
        bool(search_config.require_sl_gt_tp) if search_config is not None else True
    )
    support_ok = float(total_blocked >= min_blocked_trainable)
    useful_ok = float(blocked_bad > blocked_good) if require_sl_gt_tp else 1.0
    goal_zone_ok = float(
        (bad_block_rate >= target_sl_capture_model) and (good_block_tax <= max_tp_tax_model)
    )
    support_scale = float(max(min_blocked_trainable, 1))
    sl_target_scale = float(max(target_sl_capture_model, 1e-9))
    tp_target_scale = float(max(max_tp_tax_model, 1e-9))
    model_zone_distance = (
        max(float(min_blocked_trainable) - float(total_blocked), 0.0) / support_scale
        + max(target_sl_capture_model - bad_block_rate, 0.0) / sl_target_scale
        + max(good_block_tax - max_tp_tax_model, 0.0) / tp_target_scale
        + ((1.0 - useful_ok) if require_sl_gt_tp else 0.0)
    )
    eval_window_days = _resolve_eval_window_days(
        window_start=window_start, window_end=window_end, window_days=window_days
    )
    signals_per_30d_after = _compute_signals_per_30d_after(total_kept, eval_window_days)
    density_penalty = _compute_density_penalty(signals_per_30d_after)
    selection_score = (
        -model_zone_distance
        + (bad_block_rate - good_block_tax)
        + 0.25 * blocked_bad_precision
        + 0.0001 * float(total_blocked)
        - 0.10 * density_penalty
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
        "support_ok": float(support_ok),
        "useful_ok": float(useful_ok),
        "goal_zone_ok": float(goal_zone_ok),
        "model_zone_distance": float(model_zone_distance),
        "selection_score": float(selection_score),
    }


def sweep_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    base_block_threshold: float,
    search_config: GateThresholdSearchConfig | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in build_gate_threshold_grid(
        base_block_threshold, search_config, scored_signals_df=scored_signals_df
    ):
        decision_df, _ = apply_gate_block_threshold(scored_signals_df, threshold)
        metrics = build_gate_threshold_metrics(
            decision_df,
            search_config=search_config,
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
    search_config: GateThresholdSearchConfig | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> tuple[float, pd.DataFrame]:
    sweep_df = sweep_gate_block_threshold(
        scored_signals_df,
        base_block_threshold,
        search_config=search_config,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
    )
    if sweep_df.empty:
        raise ValueError("gate threshold sweep returned no candidates")
    ranked = sweep_df.sort_values(
        by=[
            "model_zone_distance",
            "selection_score",
            "goal_zone_ok",
            "support_ok",
            "useful_ok",
            "blocked_bad_precision",
            "good_keep_rate",
            "density_penalty",
            "block_threshold",
        ],
        ascending=[True, False, False, False, False, False, False, True, True],
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
            "counterfactual_trade_outcome",
        ),
        "scored_signals_df",
    )
    frame = scored_signals_df.copy()
    if "gate_trainable_signal" in frame.columns:
        frame = frame[frame["gate_trainable_signal"].astype(bool)].copy()
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
    frame["counterfactual_trade_outcome"] = (
        frame["counterfactual_trade_outcome"].astype(str).str.strip().str.lower()
    )
    frame = frame[frame["counterfactual_trade_outcome"].isin(["tp", "sl"])].copy()
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
        outcome = gdf["counterfactual_trade_outcome"]
        tp_rate = float(outcome.eq("tp").mean()) if len(gdf) > 0 else 0.0
        sl_rate = float(outcome.eq("sl").mean()) if len(gdf) > 0 else 0.0
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
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: object | None = None,
    execution_market_view: object | None = None,
    search_config: GateThresholdSearchConfig | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> tuple[float | None, pd.DataFrame]:
    sweep_started = time.perf_counter()
    counterfactual_columns = {
        "counterfactual_execution_status",
        "counterfactual_trade_outcome",
        "counterfactual_trade_pnl_pct",
        "counterfactual_exit_time",
    }
    if counterfactual_columns.issubset(scored_signals_df.columns):
        counterfactual_outcomes_df = scored_signals_df.loc[
            :, ["signal_id", *sorted(counterfactual_columns)]
        ].copy()
    else:
        counterfactual_outcomes_df = attach_counterfactual_execution_outcomes(
            scored_signals_df,
            bars_15m_df,
            bars_1m_df,
            execution_contract,
            bars_1s_fetcher,
            split_label="val",
        )
    independent_outcomes_df = replay_independent_short_signals(
        decision_df=scored_signals_df,
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
        execution_contract=execution_contract,
        bars_1s_fetcher=bars_1s_fetcher,
        market_view=execution_market_view,
    )
    thresholds = build_gate_threshold_grid(
        base_block_threshold, search_config, scored_signals_df=scored_signals_df
    )
    include_disabled_fallback = bool(
        search_config is not None and bool(search_config.include_disabled_candidate)
    )
    log_info(
        "GATE",
        f"threshold execution sweep start candidates_total={len(thresholds)}",
    )
    rows: list[dict[str, float]] = []
    best_selection_score = float("-inf")
    total_thresholds = len(thresholds)
    for idx, threshold in enumerate(thresholds, start=1):
        gate_mode = "threshold"
        if gate_mode == "disabled":
            gate_decisions_df = scored_signals_df.copy()
            gate_decisions_df["gate_decision"] = "keep"
            gate_decisions_df["gate_block_threshold"] = pd.NA
        else:
            gate_decisions_df, _ = apply_gate_block_threshold(
                scored_signals_df, float(threshold)
            )
        execution_decisions_df, executed_signals_df = (
            replay_short_signals_with_symbol_lock_precomputed(
                decision_df=gate_decisions_df,
                independent_outcomes_df=independent_outcomes_df,
                emit_summary_log=False,
            )
        )
        if not {
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
        }.issubset(execution_decisions_df.columns):
            execution_decisions_df = execution_decisions_df.merge(
                counterfactual_outcomes_df,
                on="signal_id",
                how="left",
                validate="one_to_one",
            )
        window_6h = build_execution_window_report(executed_signals_df, 6)
        window_24h = build_execution_window_report(executed_signals_df, 24)
        execution_metrics = build_execution_metrics(
            executed_signals_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
            precomputed_window_reports={6: window_6h, 24: window_24h},
        )
        summary = build_gate_execution_decision_summary(execution_decisions_df)
        trainable_mask = pd.Series(True, index=gate_decisions_df.index)
        if "gate_trainable_signal" in gate_decisions_df.columns:
            trainable_mask = (
                pd.to_numeric(gate_decisions_df["gate_trainable_signal"], errors="coerce")
                .fillna(0)
                .astype(int)
                .eq(1)
            )
        blocked_by_model_trainable = int(
            (
                (gate_decisions_df["gate_decision"].astype(str) == "block")
                & trainable_mask
            ).sum()
        )
        blocked_model_total = int(summary["tp_blocked_model"] + summary["sl_blocked_model"])
        blocked_sl_precision_model = _safe_ratio(
            int(summary["sl_blocked_model"]), blocked_model_total
        )
        signals_per_30d = float(execution_metrics.get("signals_per_30d", 0.0))
        pnl_sum = float(execution_metrics.get("pnl_sum", 0.0))
        worst_6h = float(window_6h["pnl_sum"].min()) if not window_6h.empty else 0.0
        worst_24h = float(window_24h["pnl_sum"].min()) if not window_24h.empty else 0.0
        max_losing_streak = float(execution_metrics.get("max_losing_streak", 0.0))
        density_penalty = _compute_density_penalty(signals_per_30d)
        selection_score = (
            10.0 * float(summary["sl_capture_model"])
            - 10.0 * float(summary["tp_tax_model"])
            + 0.50 * float(blocked_sl_precision_model)
            + 0.001 * float(blocked_by_model_trainable)
            + 0.10 * pnl_sum
            + 0.03 * worst_6h
            + 0.01 * worst_24h
            - 0.02 * max_losing_streak
            - 0.10 * density_penalty
        )
        best_selection_score = max(best_selection_score, float(selection_score))
        elapsed = time.perf_counter() - sweep_started
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (total_thresholds - idx) / rate if rate > 0 else 0.0
        threshold_repr = (
            f"{float(threshold):.6f}" if threshold is not None else "nan"
        )
        log_info(
            "GATE",
            (
                f"threshold execution progress idx={idx}/{total_thresholds} "
                f"gate_mode={gate_mode} "
                f"block_threshold={threshold_repr} "
                f"signals_after_model={int(summary['after_model'])} "
                f"signals_after_execution={int(summary['after_execution'])} "
                f"pnl_after_execution={pnl_sum:.6f} "
                f"worst_24h_after_execution={worst_24h:.6f} "
                f"tp_tax_execution={float(summary['tp_tax_execution']):.6f} "
                f"sl_capture_execution={float(summary['sl_capture_execution']):.6f} "
                f"elapsed_sec={elapsed:.3f} eta_sec={eta:.3f} "
                f"current_best={best_selection_score:.6f}"
            ),
        )
        rows.append(
            {
                "gate_mode": gate_mode,
                "block_threshold": (
                    float(threshold) if threshold is not None else float("nan")
                ),
                "signals_before": float(summary["candidate_signals_before"]),
                "signals_after_model": float(summary["after_model"]),
                "signals_after_execution": float(summary["after_execution"]),
                "blocked_by_model": float(summary["blocked_by_model"]),
                "blocked_by_model_trainable": float(blocked_by_model_trainable),
                "blocked_by_symbol_lock": float(summary["blocked_by_symbol_lock"]),
                "failed_execution_other": float(summary["failed_execution_other"]),
                "signals_per_30d_after_execution": signals_per_30d,
                "pnl_after_execution": pnl_sum,
                "worst_6h_after_execution": worst_6h,
                "worst_24h_after_execution": worst_24h,
                "max_losing_streak_after_execution": max_losing_streak,
                "tp_tax_model": float(summary["tp_tax_model"]),
                "sl_capture_model": float(summary["sl_capture_model"]),
                "blocked_sl_precision_model": float(blocked_sl_precision_model),
                "tp_blocked_model": float(summary["tp_blocked_model"]),
                "sl_blocked_model": float(summary["sl_blocked_model"]),
                "model_useful": float(
                    float(summary["sl_blocked_model"])
                    > float(summary["tp_blocked_model"])
                ),
                "model_zone_distance": 0.0,
                "tp_tax_execution": float(summary["tp_tax_execution"]),
                "sl_capture_execution": float(summary["sl_capture_execution"]),
                "model_frontier_admissible": 0.0,
                "model_frontier_dominated": 0.0,
                "model_frontier_rank": 0.0,
                "density_penalty": float(density_penalty),
                "selection_score": float(selection_score),
            }
        )
    sweep_df = pd.DataFrame(rows, columns=list(_EXECUTION_SWEEP_COLUMNS))
    if sweep_df.empty:
        raise ValueError("execution-aware gate threshold sweep returned no candidates")
    target_sl_capture_model = (
        float(search_config.target_sl_capture_model)
        if search_config is not None
        else 0.20
    )
    max_tp_tax_model = (
        float(search_config.max_tp_tax_model) if search_config is not None else 0.10
    )
    min_blocked_trainable = (
        int(search_config.min_blocked_trainable) if search_config is not None else 10
    )
    require_sl_gt_tp = (
        bool(search_config.require_sl_gt_tp) if search_config is not None else True
    )
    model_stage_df = sweep_df.copy()
    support_ok_mask = (
        model_stage_df["blocked_by_model_trainable"] >= float(min_blocked_trainable)
    )
    if require_sl_gt_tp:
        useful_mask = model_stage_df["model_useful"] > 0.5
    else:
        useful_mask = pd.Series(True, index=model_stage_df.index)
    goal_zone_mask = (
        (model_stage_df["sl_capture_model"] >= target_sl_capture_model)
        & (model_stage_df["tp_tax_model"] <= max_tp_tax_model)
    )
    primary_mask = support_ok_mask & useful_mask & goal_zone_mask
    dominated_mask = pd.Series(False, index=model_stage_df.index)
    epsilon = 1e-12
    candidate_indices = list(model_stage_df.index[primary_mask])
    for row_idx in candidate_indices:
        current = model_stage_df.loc[row_idx]
        peer_indices = [idx for idx in candidate_indices if idx != row_idx]
        if not peer_indices:
            continue
        peers = model_stage_df.loc[peer_indices]
        no_worse = (
            (peers["tp_tax_model"] <= float(current["tp_tax_model"]) + epsilon)
            & (peers["sl_capture_model"] >= float(current["sl_capture_model"]) - epsilon)
            & (
                peers["blocked_by_model_trainable"]
                >= float(current["blocked_by_model_trainable"]) - epsilon
            )
        )
        strictly_better = (
            (peers["tp_tax_model"] < float(current["tp_tax_model"]) - epsilon)
            | (peers["sl_capture_model"] > float(current["sl_capture_model"]) + epsilon)
            | (
                peers["blocked_by_model_trainable"]
                > float(current["blocked_by_model_trainable"]) + epsilon
            )
        )
        dominated_mask.loc[row_idx] = bool((no_worse & strictly_better).any())
    model_frontier_admissible = primary_mask & ~dominated_mask
    model_frontier_rank = (
        model_stage_df["sl_capture_model"]
        - model_stage_df["tp_tax_model"]
        + 0.25 * model_stage_df["blocked_sl_precision_model"]
        + 0.0001 * model_stage_df["blocked_by_model_trainable"]
    )
    support_scale = float(max(min_blocked_trainable, 1))
    sl_target_scale = float(max(target_sl_capture_model, 1e-9))
    tp_target_scale = float(max(max_tp_tax_model, 1e-9))
    support_shortfall = (
        (float(min_blocked_trainable) - model_stage_df["blocked_by_model_trainable"])
        .clip(lower=0.0)
        .astype(float)
        / support_scale
    )
    sl_shortfall = (
        (target_sl_capture_model - model_stage_df["sl_capture_model"])
        .clip(lower=0.0)
        .astype(float)
        / sl_target_scale
    )
    tp_excess = (
        (model_stage_df["tp_tax_model"] - max_tp_tax_model)
        .clip(lower=0.0)
        .astype(float)
        / tp_target_scale
    )
    if require_sl_gt_tp:
        utility_shortfall = (1.0 - model_stage_df["model_useful"]).clip(lower=0.0)
    else:
        utility_shortfall = pd.Series(0.0, index=model_stage_df.index)
    model_zone_distance = (
        support_shortfall + sl_shortfall + tp_excess + utility_shortfall
    )
    sweep_df["model_frontier_admissible"] = model_frontier_admissible.astype(float)
    sweep_df["model_frontier_dominated"] = dominated_mask.astype(float)
    sweep_df["model_frontier_rank"] = model_frontier_rank.astype(float)
    sweep_df["model_zone_distance"] = model_zone_distance.astype(float)
    if include_disabled_fallback:
        disabled_decisions_df = scored_signals_df.copy()
        disabled_decisions_df["gate_decision"] = "keep"
        disabled_decisions_df["gate_block_threshold"] = pd.NA
        execution_decisions_df, executed_signals_df = (
            replay_short_signals_with_symbol_lock_precomputed(
                decision_df=disabled_decisions_df,
                independent_outcomes_df=independent_outcomes_df,
                emit_summary_log=False,
            )
        )
        if not {
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
        }.issubset(execution_decisions_df.columns):
            execution_decisions_df = execution_decisions_df.merge(
                counterfactual_outcomes_df,
                on="signal_id",
                how="left",
                validate="one_to_one",
            )
        window_6h = build_execution_window_report(executed_signals_df, 6)
        window_24h = build_execution_window_report(executed_signals_df, 24)
        execution_metrics = build_execution_metrics(
            executed_signals_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
            precomputed_window_reports={6: window_6h, 24: window_24h},
        )
        summary = build_gate_execution_decision_summary(execution_decisions_df)
        signals_per_30d = float(execution_metrics.get("signals_per_30d", 0.0))
        pnl_sum = float(execution_metrics.get("pnl_sum", 0.0))
        worst_6h = float(window_6h["pnl_sum"].min()) if not window_6h.empty else 0.0
        worst_24h = float(window_24h["pnl_sum"].min()) if not window_24h.empty else 0.0
        max_losing_streak = float(execution_metrics.get("max_losing_streak", 0.0))
        density_penalty = _compute_density_penalty(signals_per_30d)
        blocked_model_total = int(summary["tp_blocked_model"] + summary["sl_blocked_model"])
        blocked_sl_precision_model = _safe_ratio(
            int(summary["sl_blocked_model"]), blocked_model_total
        )
        disabled_selection_score = (
            10.0 * float(summary["sl_capture_model"])
            - 10.0 * float(summary["tp_tax_model"])
            + 0.50 * float(blocked_sl_precision_model)
            + 0.10 * pnl_sum
            + 0.03 * worst_6h
            + 0.01 * worst_24h
            - 0.02 * max_losing_streak
            - 0.10 * density_penalty
        )
        disabled_model_useful = float(
            float(summary["sl_blocked_model"]) > float(summary["tp_blocked_model"])
        )
        disabled_zone_distance = (
            max(float(min_blocked_trainable) - 0.0, 0.0) / support_scale
            + max(target_sl_capture_model - float(summary["sl_capture_model"]), 0.0)
            / sl_target_scale
            + max(float(summary["tp_tax_model"]) - max_tp_tax_model, 0.0)
            / tp_target_scale
            + (
                (1.0 - disabled_model_useful)
                if require_sl_gt_tp
                else 0.0
            )
        )
        sweep_df = pd.concat(
            [
                sweep_df,
                pd.DataFrame(
                    [
                        {
                            "gate_mode": "disabled",
                            "block_threshold": float("nan"),
                            "signals_before": float(summary["candidate_signals_before"]),
                            "signals_after_model": float(summary["after_model"]),
                            "signals_after_execution": float(summary["after_execution"]),
                            "blocked_by_model": float(summary["blocked_by_model"]),
                            "blocked_by_model_trainable": 0.0,
                            "blocked_by_symbol_lock": float(summary["blocked_by_symbol_lock"]),
                            "failed_execution_other": float(summary["failed_execution_other"]),
                            "signals_per_30d_after_execution": signals_per_30d,
                            "pnl_after_execution": pnl_sum,
                            "worst_6h_after_execution": worst_6h,
                            "worst_24h_after_execution": worst_24h,
                            "max_losing_streak_after_execution": max_losing_streak,
                            "tp_tax_model": float(summary["tp_tax_model"]),
                            "sl_capture_model": float(summary["sl_capture_model"]),
                            "blocked_sl_precision_model": float(blocked_sl_precision_model),
                            "tp_blocked_model": float(summary["tp_blocked_model"]),
                            "sl_blocked_model": float(summary["sl_blocked_model"]),
                            "model_useful": disabled_model_useful,
                            "model_zone_distance": float(disabled_zone_distance),
                            "tp_tax_execution": float(summary["tp_tax_execution"]),
                            "sl_capture_execution": float(summary["sl_capture_execution"]),
                            "model_frontier_admissible": 0.0,
                            "model_frontier_dominated": 0.0,
                            "model_frontier_rank": 0.0,
                            "density_penalty": float(density_penalty),
                            "selection_score": float(disabled_selection_score),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    best_mode = "threshold"
    best_threshold: float | None = None
    selected_row: pd.Series | None = None
    if bool(model_frontier_admissible.any()):
        ranked = sweep_df.loc[sweep_df["model_frontier_admissible"] > 0.5].sort_values(
            by=[
                "model_frontier_rank",
                "sl_capture_model",
                "tp_tax_model",
                "blocked_sl_precision_model",
                "blocked_by_model_trainable",
                "selection_score",
                "pnl_after_execution",
                "worst_24h_after_execution",
                "worst_6h_after_execution",
                "max_losing_streak_after_execution",
                "block_threshold",
            ],
            ascending=[
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        best_mode = str(ranked.iloc[0]["gate_mode"])
        best_threshold = (
            None if best_mode == "disabled" else float(ranked.iloc[0]["block_threshold"])
        )
        selected_row = ranked.iloc[0]
    else:
        ranked = sweep_df.sort_values(
            by=[
                "model_zone_distance",
                "model_frontier_rank",
                "sl_capture_model",
                "tp_tax_model",
                "blocked_sl_precision_model",
                "blocked_by_model_trainable",
                "selection_score",
                "pnl_after_execution",
                "worst_24h_after_execution",
                "worst_6h_after_execution",
                "max_losing_streak_after_execution",
                "block_threshold",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        best_mode = str(ranked.iloc[0]["gate_mode"])
        best_threshold = (
            None if best_mode == "disabled" else float(ranked.iloc[0]["block_threshold"])
        )
        selected_row = ranked.iloc[0]
    log_info(
        "GATE",
        (
            "threshold model-frontier summary "
            f"support_ok={int(support_ok_mask.sum())} "
            f"useful_ok={int(useful_mask.sum())} "
            f"goal_zone_ok={int(goal_zone_mask.sum())} "
            f"primary_ok={int(primary_mask.sum())} "
            f"frontier_admissible={int(model_frontier_admissible.sum())} "
            f"target_sl_capture_model={target_sl_capture_model:.6f} "
            f"max_tp_tax_model={max_tp_tax_model:.6f} "
            f"min_blocked_trainable={int(min_blocked_trainable)} "
            f"require_sl_gt_tp={int(require_sl_gt_tp)}"
        ),
    )
    log_info(
        "GATE",
        f"threshold execution sweep done candidates_total={len(sweep_df)} elapsed_sec={time.perf_counter() - sweep_started:.3f}",
    )
    log_info(
        "GATE",
        (
            "gate threshold execution-aware select done "
            f"best_mode={best_mode} "
            f"best_threshold={(f'{best_threshold:.6f}' if best_threshold is not None else 'None')} "
            f"best_selection_score={float(selected_row['selection_score']) if selected_row is not None else float('nan'):.6f}"
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


def _build_empirical_threshold_grid(
    scored_signals_df: pd.DataFrame | None,
) -> list[float]:
    if scored_signals_df is None or scored_signals_df.empty:
        return []
    if "p_block" not in scored_signals_df.columns:
        return []
    p_block = pd.to_numeric(scored_signals_df["p_block"], errors="coerce").dropna()
    if p_block.empty:
        return []
    clipped = p_block.astype(float).clip(lower=0.0, upper=1.0)
    unique_scores = sorted(set(float(value) for value in clipped.tolist()))
    if not unique_scores:
        return []
    tail_start = int(len(unique_scores) * 0.60)
    tail_scores = unique_scores[tail_start:]
    if len(tail_scores) > 64:
        positions = {
            int(round(i * (len(tail_scores) - 1) / 63))
            for i in range(64)
        }
        tail_scores = [tail_scores[pos] for pos in sorted(positions)]
    quantile_scores = [
        float(value)
        for value in clipped.quantile([0.70, 0.80, 0.90, 0.95, 0.975, 0.99]).tolist()
    ]
    out = {_round6(_clip(value, 0.0, 1.0)) for value in [*tail_scores, *quantile_scores]}
    return sorted(out)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def attach_counterfactual_execution_outcomes(
    decision_df: pd.DataFrame,
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: object | None = None,
    split_label: str = "unknown",
    execution_market_view: object | None = None,
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
    total_signals = len(decision_df)
    started = time.perf_counter()
    progress_iter = tqdm(
        [0],
        total=1,
        desc=f"counterfactual {split_label}",
        unit="batch",
        **_tqdm_kwargs(),
    )
    independent_outcomes_df = replay_independent_short_signals(
        decision_df=decision_df,
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
        execution_contract=execution_contract,
        bars_1s_fetcher=bars_1s_fetcher,
        market_view=execution_market_view,
    )
    for _ in progress_iter:
        pass
    outcome_counts = (
        independent_outcomes_df["trade_outcome"]
        .astype(str)
        .str.lower()
        .value_counts(dropna=False)
    )
    status_counts: dict[str, int] = {
        "tp": int(outcome_counts.get("tp", 0)),
        "sl": int(outcome_counts.get("sl", 0)),
        "timeout": int(outcome_counts.get("timeout", 0)),
        "ambiguous": int(outcome_counts.get("ambiguous", 0)),
    }
    elapsed_total = time.perf_counter() - started
    avg_sec_per_signal = elapsed_total / total_signals if total_signals > 0 else 0.0
    log_info(
        "EXECUTION",
        (
            f"counterfactual done split={split_label} signals_total={total_signals} "
            f"elapsed_sec={elapsed_total:.3f} avg_sec_per_signal={avg_sec_per_signal:.3f} "
            f"tp_total={status_counts['tp']} sl_total={status_counts['sl']} "
            f"timeout_total={status_counts['timeout']} ambiguous_total={status_counts['ambiguous']}"
        ),
    )
    out = independent_outcomes_df.rename(
        columns={
            "execution_status": "counterfactual_execution_status",
            "trade_outcome": "counterfactual_trade_outcome",
            "trade_pnl_pct": "counterfactual_trade_pnl_pct",
            "exit_time": "counterfactual_exit_time",
        }
    ).loc[:, list(_COUNTERFACTUAL_COLUMNS)]
    if not decision_df["signal_id"].is_unique:
        raise ValueError(
            "decision_df must have unique signal_id for counterfactual enrichment"
        )
    if not out["signal_id"].is_unique:
        raise ValueError("counterfactual outcomes produced duplicate signal_id")
    return out
