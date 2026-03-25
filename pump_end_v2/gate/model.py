from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import GateModelConfig
from pump_end_v2.gate.feature_view import GATE_FEATURE_COLUMNS, GATE_IDENTITY_COLUMNS
from pump_end_v2.logging import log_info

_VAL_SCORED_COLUMNS: tuple[str, ...] = (
    *GATE_IDENTITY_COLUMNS,
    "p_block",
    "target_block_signal",
    "block_reason",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
)


def build_gate_model(model_config: GateModelConfig) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
        iterations=model_config.iterations,
        depth=model_config.depth,
        learning_rate=model_config.learning_rate,
        l2_leaf_reg=model_config.l2_leaf_reg,
        random_seed=model_config.random_seed,
    )


def fit_gate_model(
    model: CatBoostClassifier,
    train_df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
) -> CatBoostClassifier:
    _require_columns(train_df, [*feature_columns, target_column], "train_df")
    x_train = train_df.loc[:, list(feature_columns)]
    y_train = train_df[target_column].astype(int)
    model.fit(x_train, y_train)
    return model


def predict_gate_scores(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    _require_columns(df, [*GATE_IDENTITY_COLUMNS, *feature_columns], "df")
    if df.empty:
        return pd.DataFrame(columns=[*GATE_IDENTITY_COLUMNS, "p_block"])
    proba = model.predict_proba(df.loc[:, list(feature_columns)])[:, 1]
    out = df.loc[:, list(GATE_IDENTITY_COLUMNS)].copy()
    out["p_block"] = proba.astype(float)
    return out


def fit_gate_on_train_oof_and_score_val(
    train_gate_dataset_df: pd.DataFrame,
    val_gate_dataset_df: pd.DataFrame,
    gate_model_config: GateModelConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    _require_columns(
        train_gate_dataset_df,
        [*GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS, "target_block_signal", "gate_trainable_signal"],
        "train_gate_dataset_df",
    )
    _require_columns(
        val_gate_dataset_df,
        [*GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS, "target_block_signal", "block_reason", "target_good_short_now", "target_reason", "future_outcome_class", "future_prepullback_squeeze_pct", "future_pullback_pct", "future_net_edge_pct"],
        "val_gate_dataset_df",
    )
    train_fit = train_gate_dataset_df[
        (train_gate_dataset_df["gate_trainable_signal"].astype(bool))
        & (train_gate_dataset_df["score_source"] == "train_oof")
    ].copy()
    if train_fit.empty:
        raise ValueError("no train rows for gate model fit after filters")
    val_score = val_gate_dataset_df[val_gate_dataset_df["score_source"] == "val_forward"].copy()
    if val_score.empty:
        raise ValueError("no val_forward rows for gate model scoring")
    model = build_gate_model(gate_model_config)
    fit_gate_model(model, train_fit, GATE_FEATURE_COLUMNS, "target_block_signal")
    score_df = predict_gate_scores(model, val_score, GATE_FEATURE_COLUMNS)
    scored = val_score.merge(score_df[["signal_id", "p_block"]], on="signal_id", how="left", validate="one_to_one")
    out = scored.loc[:, list(_VAL_SCORED_COLUMNS)].copy()
    log_info(
        "GATE",
        f"gate model val scoring done train_rows={len(train_fit)} val_rows={len(out)}",
    )
    return model, out.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
