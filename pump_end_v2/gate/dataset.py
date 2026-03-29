import pandas as pd

from pump_end_v2.gate.feature_view import GATE_FEATURE_COLUMNS, GATE_IDENTITY_COLUMNS
from pump_end_v2.logging import log_info

GATE_TARGET_META_COLUMNS: tuple[str, ...] = (
    "target_block_signal",
    "block_reason",
    "counterfactual_trade_outcome",
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
    "counterfactual_trade_outcome",
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


def build_gate_dataset(
    gate_feature_view_df: pd.DataFrame, candidate_signals_df: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(
        gate_feature_view_df,
        ("signal_id", *GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS),
        "gate_feature_view_df",
    )
    _require_columns(
        candidate_signals_df, _CANDIDATE_TARGET_COLUMNS, "candidate_signals_df"
    )
    _require_unique_signal_id(gate_feature_view_df, "gate_feature_view_df")
    _require_unique_signal_id(candidate_signals_df, "candidate_signals_df")
    _validate_no_leakage_columns()
    target_part = candidate_signals_df.loc[:, list(_CANDIDATE_TARGET_COLUMNS)].copy()
    merged = gate_feature_view_df.merge(
        target_part, on="signal_id", how="inner", validate="one_to_one"
    )
    merged["target_good_short_now"] = (
        pd.to_numeric(merged["target_good_short_now"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    merged["counterfactual_trade_outcome"] = (
        merged["counterfactual_trade_outcome"].astype(str).str.strip().str.lower()
    )
    merged["signal_quality_h32"] = merged["signal_quality_h32"].astype(str)
    merged["target_block_signal"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    sl_mask = merged["counterfactual_trade_outcome"].eq("sl")
    tp_mask = merged["counterfactual_trade_outcome"].eq("tp")
    merged.loc[sl_mask, "target_block_signal"] = 1
    merged.loc[tp_mask, "target_block_signal"] = 0
    merged["target_reason"] = merged["target_reason"].astype(str)
    merged["future_outcome_class"] = merged["future_outcome_class"].astype(str)
    merged["gate_trainable_signal"] = sl_mask | tp_mask
    merged["block_reason"] = merged["counterfactual_trade_outcome"].apply(
        _resolve_block_reason
    )
    ordered = merged.loc[
        :,
        [
            *GATE_IDENTITY_COLUMNS,
            *GATE_FEATURE_COLUMNS,
            *GATE_TARGET_META_COLUMNS,
            "gate_trainable_signal",
        ],
    ]
    trainable_rows = int(ordered["gate_trainable_signal"].sum())
    positive_rate = (
        float(pd.to_numeric(ordered["target_block_signal"], errors="coerce").mean())
        if len(ordered) > 0
        else 0.0
    )
    log_info(
        "GATE",
        (
            "gate dataset build done "
            f"rows_total={len(ordered)} trainable_rows={trainable_rows} positive_rate={positive_rate:.6f}"
        ),
    )
    return ordered.reset_index(drop=True)


def _resolve_block_reason(outcome_value: object) -> str:
    outcome = str(outcome_value).strip().lower()
    if outcome == "sl":
        return "block_sl"
    if outcome == "tp":
        return "keep_tp"
    if outcome == "timeout":
        return "skip_timeout"
    if outcome == "ambiguous":
        return "skip_ambiguous"
    return "skip_unknown"


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
