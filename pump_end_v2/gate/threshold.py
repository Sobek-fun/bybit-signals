from __future__ import annotations

import pandas as pd

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
    "p_block",
    "context_bar_open_time",
    "target_block_signal",
    "block_reason",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
)


def build_gate_threshold_grid(base_block_threshold: float) -> list[float]:
    base = float(base_block_threshold)
    raw = [base - 0.20, base - 0.10, base - 0.05, base, base + 0.05, base + 0.10, base + 0.20]
    clipped = [_round6(_clip(value, 0.05, 0.95)) for value in raw]
    return sorted(set(clipped))


def apply_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    block_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(scored_signals_df, _APPLY_REQUIRED_COLUMNS, "scored_signals_df")
    threshold = float(block_threshold)
    decision_df = scored_signals_df.copy()
    decision_df["gate_decision"] = decision_df["p_block"].astype(float).ge(threshold).map({True: "block", False: "keep"})
    decision_df["gate_block_threshold"] = threshold
    kept_signals_df = decision_df[decision_df["gate_decision"] == "keep"].copy()
    return decision_df, kept_signals_df


def build_gate_threshold_metrics(decision_df: pd.DataFrame) -> dict[str, float]:
    _require_columns(decision_df, ("target_block_signal", "gate_decision", "context_bar_open_time"), "decision_df")
    eval_df = decision_df.copy()
    if "gate_trainable_signal" in eval_df.columns:
        eval_df = eval_df[eval_df["gate_trainable_signal"].astype(bool)].copy()
    signals_before = int(len(eval_df))
    blocked_mask = eval_df["gate_decision"] == "block"
    kept_mask = eval_df["gate_decision"] == "keep"
    bad_mask = pd.to_numeric(eval_df["target_block_signal"], errors="coerce").fillna(0).astype(int) == 1
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
    signals_per_30d_after = _compute_signals_per_30d_after(eval_df[kept_mask].copy())
    density_penalty = _compute_density_penalty(signals_per_30d_after)
    selection_score = (
        bad_block_rate + good_keep_rate + blocked_bad_precision - good_block_tax - 0.25 * density_penalty
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
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in build_gate_threshold_grid(base_block_threshold):
        decision_df, _ = apply_gate_block_threshold(scored_signals_df, threshold)
        metrics = build_gate_threshold_metrics(decision_df)
        rows.append({"block_threshold": threshold, **metrics})
    out = pd.DataFrame(rows, columns=list(_SWEEP_COLUMNS))
    log_info("GATE", f"gate threshold sweep done candidates_total={len(out)}")
    return out


def select_gate_block_threshold(
    scored_signals_df: pd.DataFrame,
    base_block_threshold: float,
) -> tuple[float, pd.DataFrame]:
    sweep_df = sweep_gate_block_threshold(scored_signals_df, base_block_threshold)
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


def _compute_signals_per_30d_after(kept_df: pd.DataFrame) -> float:
    if kept_df.empty:
        return 0.0
    context = pd.to_datetime(kept_df["context_bar_open_time"], utc=True, errors="coerce").dropna()
    if context.empty:
        return 0.0
    days_span = (context.max() - context.min()) / pd.Timedelta(days=1)
    safe_days_span = max(float(days_span), 1.0)
    return float(len(context) * 30.0 / safe_days_span)


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


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
