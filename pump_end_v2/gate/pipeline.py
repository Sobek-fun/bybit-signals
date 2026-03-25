from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import GateModelConfig
from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.gate.dataset import GATE_TARGET_META_COLUMNS, build_gate_dataset
from pump_end_v2.gate.feature_view import GATE_FEATURE_COLUMNS, GATE_IDENTITY_COLUMNS, build_gate_feature_view
from pump_end_v2.gate.model import build_gate_model, fit_gate_model, predict_gate_scores
from pump_end_v2.gate.threshold import sweep_gate_block_threshold
from pump_end_v2.logging import log_info

_SCORED_OUTPUT_COLUMNS: tuple[str, ...] = (
    *GATE_IDENTITY_COLUMNS,
    "p_block",
    "target_block_signal",
    "block_reason",
    "signal_quality_h32",
    "gate_trainable_signal",
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


def build_gate_val_scored_signals_and_datasets(
    train_oof_candidate_signals_df: pd.DataFrame,
    val_candidate_signals_df: pd.DataFrame,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    gate_model_config: GateModelConfig,
    base_block_threshold: float,
) -> tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_feature_view_df = build_gate_feature_view(
        candidate_signals_df=train_oof_candidate_signals_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
        execution_contract=execution_contract,
    )
    val_feature_view_df = build_gate_feature_view(
        candidate_signals_df=val_candidate_signals_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
        execution_contract=execution_contract,
    )
    train_gate_dataset_df = build_gate_dataset(train_feature_view_df, train_oof_candidate_signals_df)
    val_gate_dataset_df = build_gate_dataset(val_feature_view_df, val_candidate_signals_df)
    train_fit_df = train_gate_dataset_df[
        (train_gate_dataset_df["score_source"] == "train_oof")
        & train_gate_dataset_df["gate_trainable_signal"].astype(bool)
    ].copy()
    if train_fit_df.empty:
        raise ValueError("no train_oof + gate_trainable rows for gate fit")
    val_score_df = val_gate_dataset_df[val_gate_dataset_df["score_source"] == "val_forward"].copy()
    if val_score_df.empty:
        raise ValueError("no val_forward rows for gate scoring")
    model = build_gate_model(gate_model_config)
    fit_gate_model(model, train_fit_df, GATE_FEATURE_COLUMNS, "target_block_signal")
    val_scored_signals_df = _score_gate_dataset_rows(model, val_score_df)
    threshold_sweep_df = sweep_gate_block_threshold(
        scored_signals_df=val_scored_signals_df,
        base_block_threshold=base_block_threshold,
    )
    log_info(
        "GATE",
        (
            "gate val wrapper done "
            f"train_rows={len(train_fit_df)} val_rows={len(val_score_df)} scored_rows={len(val_scored_signals_df)}"
        ),
    )
    return model, val_scored_signals_df, threshold_sweep_df, train_gate_dataset_df, val_gate_dataset_df


def build_gate_test_scored_signals(
    train_oof_candidate_signals_df: pd.DataFrame,
    test_candidate_signals_df: pd.DataFrame,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    gate_model_config: GateModelConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame]:
    train_feature_view_df = build_gate_feature_view(
        candidate_signals_df=train_oof_candidate_signals_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
        execution_contract=execution_contract,
    )
    test_feature_view_df = build_gate_feature_view(
        candidate_signals_df=test_candidate_signals_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
        execution_contract=execution_contract,
    )
    train_gate_dataset_df = build_gate_dataset(train_feature_view_df, train_oof_candidate_signals_df)
    test_gate_dataset_df = build_gate_dataset(test_feature_view_df, test_candidate_signals_df)
    train_fit_df = train_gate_dataset_df[
        (train_gate_dataset_df["score_source"] == "train_oof")
        & train_gate_dataset_df["gate_trainable_signal"].astype(bool)
    ].copy()
    if train_fit_df.empty:
        raise ValueError("no train_oof + gate_trainable rows for gate fit")
    test_score_df = test_gate_dataset_df[test_gate_dataset_df["score_source"] == "test_forward"].copy()
    if test_score_df.empty:
        raise ValueError("no test_forward rows for gate scoring")
    model = build_gate_model(gate_model_config)
    fit_gate_model(model, train_fit_df, GATE_FEATURE_COLUMNS, "target_block_signal")
    test_scored_signals_df = _score_gate_dataset_rows(model, test_score_df)
    log_info(
        "GATE",
        (
            "gate test wrapper done "
            f"train_rows={len(train_fit_df)} test_rows={len(test_score_df)} scored_rows={len(test_scored_signals_df)}"
        ),
    )
    return model, test_scored_signals_df, test_gate_dataset_df


def _score_gate_dataset_rows(model: CatBoostClassifier, rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=list(_SCORED_OUTPUT_COLUMNS))
    score_df = predict_gate_scores(model, rows_df, GATE_FEATURE_COLUMNS)
    merged = rows_df.merge(score_df[["signal_id", "p_block"]], on="signal_id", how="left", validate="one_to_one")
    required_cols = [*GATE_IDENTITY_COLUMNS, *GATE_TARGET_META_COLUMNS, "target_block_signal", "gate_trainable_signal"]
    _require_columns(merged, required_cols, "scored_rows")
    return merged.loc[:, list(_SCORED_OUTPUT_COLUMNS)].reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
