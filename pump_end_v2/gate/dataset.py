from __future__ import annotations

import pandas as pd

from pump_end_v2.gate.feature_view import GATE_FEATURE_COLUMNS, GATE_IDENTITY_COLUMNS
from pump_end_v2.logging import log_info

GATE_TARGET_META_COLUMNS: tuple[str, ...] = (
    "target_block_signal",
    "block_reason",
    "signal_quality_h32",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
    "bars_to_pullback",
    "bars_to_peak_after_row",
    "bars_to_resolution",
    "entry_quality_score",
    "ideal_entry_row_id",
    "ideal_entry_bar_open_time",
    "is_ideal_entry",
)

_CANDIDATE_TARGET_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "signal_quality_h32",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
    "bars_to_pullback",
    "bars_to_peak_after_row",
    "bars_to_resolution",
    "entry_quality_score",
    "ideal_entry_row_id",
    "ideal_entry_bar_open_time",
    "is_ideal_entry",
)


def build_gate_dataset(gate_feature_view_df: pd.DataFrame, candidate_signals_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(gate_feature_view_df, ("signal_id", *GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS), "gate_feature_view_df")
    _require_columns(candidate_signals_df, _CANDIDATE_TARGET_COLUMNS, "candidate_signals_df")
    _require_unique_signal_id(gate_feature_view_df, "gate_feature_view_df")
    _require_unique_signal_id(candidate_signals_df, "candidate_signals_df")
    _validate_no_leakage_columns()
    target_part = candidate_signals_df.loc[:, list(_CANDIDATE_TARGET_COLUMNS)].copy()
    merged = gate_feature_view_df.merge(target_part, on="signal_id", how="inner", validate="one_to_one")
    merged["target_good_short_now"] = pd.to_numeric(merged["target_good_short_now"], errors="coerce").fillna(0).astype(int)
    merged["signal_quality_h32"] = merged["signal_quality_h32"].astype(str)
    merged["target_block_signal"] = (merged["signal_quality_h32"] != "clean_retrace_h32").astype(int)
    merged["target_reason"] = merged["target_reason"].astype(str)
    merged["future_outcome_class"] = merged["future_outcome_class"].astype(str)
    merged["gate_trainable_signal"] = (
        merged["signal_quality_h32"].ne("")
        & merged["signal_quality_h32"].ne("nan")
        & merged["signal_quality_h32"].ne("<NA>")
    )
    merged["block_reason"] = merged.apply(_resolve_block_reason, axis=1)
    ordered = merged.loc[:, [*GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS, *GATE_TARGET_META_COLUMNS, "gate_trainable_signal"]]
    trainable_rows = int(ordered["gate_trainable_signal"].sum())
    positive_rate = float(pd.to_numeric(ordered["target_block_signal"], errors="coerce").mean()) if len(ordered) > 0 else 0.0
    log_info(
        "GATE",
        (
            "gate dataset build done "
            f"rows_total={len(ordered)} trainable_rows={trainable_rows} positive_rate={positive_rate:.6f}"
        ),
    )
    return ordered.reset_index(drop=True)


def _resolve_block_reason(row: pd.Series) -> str:
    quality = str(row["signal_quality_h32"])
    if quality == "clean_retrace_h32":
        return "keep_clean_retrace_h32"
    if quality == "dirty_retrace_h32":
        return "block_dirty_retrace_h32"
    if quality == "clean_no_pullback_h32":
        return "block_clean_no_pullback_h32"
    if quality == "dirty_no_pullback_h32":
        return "block_dirty_no_pullback_h32"
    if quality == "pullback_before_squeeze_h32":
        return "block_pullback_before_squeeze_h32"
    return "block_unknown_quality"


def _validate_no_leakage_columns() -> None:
    leakage_columns = set(GATE_TARGET_META_COLUMNS) | {"gate_trainable_signal"}
    leaked = [column for column in GATE_FEATURE_COLUMNS if column in leakage_columns]
    if leaked:
        raise ValueError(f"leakage columns found in GATE_FEATURE_COLUMNS: {leaked}")


def _require_unique_signal_id(df: pd.DataFrame, name: str) -> None:
    if not df["signal_id"].is_unique:
        raise ValueError(f"{name} must have unique signal_id")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
