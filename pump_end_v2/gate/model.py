import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import GateModelConfig
from pump_end_v2.gate.feature_view import GATE_IDENTITY_COLUMNS


def build_gate_model(model_config: GateModelConfig) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
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
    tp_row_weight: float = 2.0,
    sl_row_weight: float = 1.0,
) -> CatBoostClassifier:
    _require_columns(train_df, [*feature_columns, target_column], "train_df")
    x_train = train_df.loc[:, list(feature_columns)]
    y_train = train_df[target_column].astype(int)
    sample_weight = (
        y_train.eq(0).astype(float) * float(tp_row_weight)
        + y_train.eq(1).astype(float) * float(sl_row_weight)
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
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


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
