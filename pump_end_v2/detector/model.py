from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import DetectorModelConfig
from pump_end_v2.features.manifest import DETECTOR_IDENTITY_COLUMNS


def build_detector_model(model_config: DetectorModelConfig) -> CatBoostClassifier:
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


def fit_detector_model(
    model: CatBoostClassifier,
    train_df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    eval_df: pd.DataFrame | None = None,
) -> CatBoostClassifier:
    _require_columns(train_df, [*feature_columns, target_column], "train_df")
    x_train = train_df.loc[:, list(feature_columns)]
    y_train = train_df[target_column].astype(int)
    if eval_df is None:
        model.fit(x_train, y_train)
        return model
    _require_columns(eval_df, [*feature_columns, target_column], "eval_df")
    x_eval = eval_df.loc[:, list(feature_columns)]
    y_eval = eval_df[target_column].astype(int)
    model.fit(x_train, y_train, eval_set=(x_eval, y_eval))
    return model


def predict_detector_scores(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    _require_columns(df, [*DETECTOR_IDENTITY_COLUMNS, *feature_columns], "df")
    if df.empty:
        return pd.DataFrame(columns=[*DETECTOR_IDENTITY_COLUMNS, "p_good"])
    scores = model.predict_proba(df.loc[:, list(feature_columns)])[:, 1]
    out = df.loc[:, list(DETECTOR_IDENTITY_COLUMNS)].copy()
    out["p_good"] = scores.astype(float)
    return out


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
