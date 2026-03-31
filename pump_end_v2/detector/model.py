import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import DetectorModelConfig
from pump_end_v2.detector.reason_classes import MODEL_REASON_CLASS_TO_ID
from pump_end_v2.features.manifest import DETECTOR_IDENTITY_COLUMNS

PROBABILITY_COLUMNS: tuple[str, ...] = (
    "p_good",
    "p_too_early",
    "p_too_late",
    "p_continuation_flat",
)

_CLASS_ID_TO_LABEL: dict[int, str] = {
    class_id: label for label, class_id in MODEL_REASON_CLASS_TO_ID.items()
}


def build_detector_model(model_config: DetectorModelConfig) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
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
        return pd.DataFrame(
            columns=[
                *DETECTOR_IDENTITY_COLUMNS,
                *PROBABILITY_COLUMNS,
                "predicted_reason_group",
                "predicted_reason_group_id",
            ]
        )
    probabilities = model.predict_proba(df.loc[:, list(feature_columns)])
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError("detector model has no classes_ after fit")
    out = df.loc[:, list(DETECTOR_IDENTITY_COLUMNS)].copy()
    for probability_column in PROBABILITY_COLUMNS:
        out[probability_column] = 0.0
    if probabilities.ndim != 2:
        raise ValueError(
            f"predict_proba must return 2D array, got ndim={probabilities.ndim}"
        )
    if probabilities.shape[1] != len(model_classes):
        raise ValueError(
            "predict_proba/classes_ mismatch "
            f"proba_cols={probabilities.shape[1]} classes={len(model_classes)}"
        )
    for class_position, raw_class_id in enumerate(model_classes):
        class_id = int(raw_class_id)
        if class_id not in _CLASS_ID_TO_LABEL:
            raise ValueError(f"unknown class id from model.classes_: {raw_class_id!r}")
        probability_column = f"p_{_CLASS_ID_TO_LABEL[class_id]}"
        out[probability_column] = probabilities[:, class_position].astype(float)
    predicted_reason_group = out.loc[:, list(PROBABILITY_COLUMNS)].idxmax(axis=1)
    out["predicted_reason_group"] = predicted_reason_group.str.removeprefix("p_")
    out["predicted_reason_group_id"] = (
        out["predicted_reason_group"].map(MODEL_REASON_CLASS_TO_ID).astype("Int64")
    )
    return out


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
