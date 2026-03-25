from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import DetectorCVConfig, DetectorModelConfig, ResolverConfig, SplitBounds
from pump_end_v2.detector.model import build_detector_model, fit_detector_model, predict_detector_scores
from pump_end_v2.detector.splits import filter_fold_rows, generate_detector_walkforward_folds
from pump_end_v2.features.manifest import DETECTOR_FEATURE_COLUMNS
from pump_end_v2.logging import log_info


def build_detector_train_oof_scores(
    dataset_df: pd.DataFrame,
    split_bounds: SplitBounds,
    resolver_config: ResolverConfig,
    detector_cv_config: DetectorCVConfig,
    detector_model_config: DetectorModelConfig,
) -> pd.DataFrame:
    _require_columns(dataset_df, ["dataset_split", "trainable_row", "target_good_short_now"])
    train_df = dataset_df[(dataset_df["dataset_split"] == "train") & (dataset_df["trainable_row"].astype(bool))].copy()
    folds = generate_detector_walkforward_folds(dataset_df, split_bounds, resolver_config, detector_cv_config)
    predictions: list[pd.DataFrame] = []
    covered_ids: set[str] = set()
    for fold in folds:
        fold_train_raw, fold_val_raw = filter_fold_rows(dataset_df, fold)
        fold_train = fold_train_raw[fold_train_raw["trainable_row"].astype(bool)].copy()
        fold_val = fold_val_raw[fold_val_raw["trainable_row"].astype(bool)].copy()
        if fold_train.empty or fold_val.empty:
            continue
        model = build_detector_model(detector_model_config)
        fit_detector_model(model, fold_train, DETECTOR_FEATURE_COLUMNS, "target_good_short_now")
        scored = predict_detector_scores(model, fold_val, DETECTOR_FEATURE_COLUMNS)
        scored = scored.merge(
            fold_val[
                ["decision_row_id", "target_good_short_now", "target_reason", "future_outcome_class"]
            ],
            on="decision_row_id",
            how="left",
            validate="one_to_one",
        )
        scored["fold_id"] = fold.fold_id
        scored["score_source"] = "train_oof"
        predictions.append(scored)
        covered_ids.update(fold_val["decision_row_id"].astype(str).tolist())
        train_positive_rate = float(fold_train["target_good_short_now"].astype(float).mean())
        val_positive_rate = float(fold_val["target_good_short_now"].astype(float).mean())
        log_info(
            "CV",
            (
                f"{fold.fold_id} train_rows={len(fold_train)} val_rows={len(fold_val)} "
                f"train_positive_rate={train_positive_rate:.4f} val_positive_rate={val_positive_rate:.4f}"
            ),
        )
    if predictions:
        oof = pd.concat(predictions, ignore_index=True)
    else:
        oof = pd.DataFrame(
            columns=[
                "decision_row_id",
                "episode_id",
                "symbol",
                "context_bar_open_time",
                "decision_time",
                "entry_bar_open_time",
                "p_good",
                "target_good_short_now",
                "target_reason",
                "future_outcome_class",
                "fold_id",
                "score_source",
            ]
        )
    dup_mask = oof["decision_row_id"].duplicated(keep=False)
    if dup_mask.any():
        duplicates = oof.loc[dup_mask, "decision_row_id"].astype(str).head(5).tolist()
        raise ValueError(f"OOF predictions contain duplicate decision_row_id sample={duplicates}")
    coverage_denominator = len(train_df[train_df["decision_row_id"].astype(str).isin(covered_ids)])
    coverage = float(len(oof) / coverage_denominator) if coverage_denominator > 0 else 0.0
    log_info("DETECTOR", f"OOF done rows_total={len(oof)} coverage={coverage:.4f}")
    ordered = [
        "decision_row_id",
        "episode_id",
        "symbol",
        "context_bar_open_time",
        "decision_time",
        "entry_bar_open_time",
        "p_good",
        "target_good_short_now",
        "target_reason",
        "future_outcome_class",
        "fold_id",
        "score_source",
    ]
    return oof.loc[:, ordered].copy()


def fit_detector_on_train_and_score_val(
    dataset_df: pd.DataFrame,
    split_bounds: SplitBounds,
    resolver_config: ResolverConfig,
    detector_model_config: DetectorModelConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    _require_columns(
        dataset_df,
        [
            "dataset_split",
            "trainable_row",
            "context_bar_open_time",
            "target_good_short_now",
            "target_reason",
            "future_outcome_class",
        ],
    )
    frame = dataset_df.copy()
    frame["context_bar_open_time"] = pd.to_datetime(frame["context_bar_open_time"], utc=True, errors="raise")
    purge_gap_timedelta = pd.Timedelta(minutes=15 * resolver_config.horizon_bars)
    effective_train_end = pd.Timestamp(split_bounds.train_end) - purge_gap_timedelta
    train_fit = frame[
        (frame["dataset_split"] == "train")
        & frame["trainable_row"].astype(bool)
        & (frame["context_bar_open_time"] <= effective_train_end)
    ].copy()
    val_rows = frame[(frame["dataset_split"] == "val") & frame["trainable_row"].astype(bool)].copy()
    if train_fit.empty:
        raise ValueError("no trainable train rows available for final detector fit")
    model = build_detector_model(detector_model_config)
    fit_detector_model(model, train_fit, DETECTOR_FEATURE_COLUMNS, "target_good_short_now")
    val_scores = predict_detector_scores(model, val_rows, DETECTOR_FEATURE_COLUMNS)
    val_scores = val_scores.merge(
        val_rows[
            ["decision_row_id", "target_good_short_now", "target_reason", "future_outcome_class"]
        ],
        on="decision_row_id",
        how="left",
        validate="one_to_one",
    )
    val_scores["score_source"] = "val_forward"
    ordered = [
        "decision_row_id",
        "episode_id",
        "symbol",
        "context_bar_open_time",
        "decision_time",
        "entry_bar_open_time",
        "p_good",
        "target_good_short_now",
        "target_reason",
        "future_outcome_class",
        "score_source",
    ]
    val_scores = val_scores.loc[:, ordered].copy()
    log_info("DETECTOR", f"final train->val scoring done train_rows={len(train_fit)} val_rows={len(val_scores)}")
    return model, val_scores


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
