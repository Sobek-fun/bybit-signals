import time

import pandas as pd

from pump_end_v2.features.manifest import (
    BLOCKED_COLUMNS,
    DETECTOR_FEATURE_COLUMNS,
    DETECTOR_IDENTITY_COLUMNS,
)
from pump_end_v2.logging import log_info, stage_done, stage_start

TARGET_META_COLUMNS: tuple[str, ...] = (
    "episode_open_time",
    "is_resolved",
    "row_trade_outcome",
    "row_trade_pnl_pct",
    "row_mfe_pct",
    "row_mae_pct",
    "row_holding_bars",
    "target_row_weight",
    "future_outcome_class",
    "signal_quality_h32",
    "target_good_short_now",
    "target_reason",
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


def build_detector_dataset(
        detector_feature_view_df: pd.DataFrame, resolved_rows_df: pd.DataFrame
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("DETECTOR", "DATASET_BUILD")
    _validate_unique_ids(detector_feature_view_df, "detector_feature_view_df")
    _validate_unique_ids(resolved_rows_df, "resolved_rows_df")
    _validate_feature_view_columns(detector_feature_view_df)
    _validate_resolved_columns(resolved_rows_df)
    resolved_cols = ["decision_row_id", *TARGET_META_COLUMNS]
    merged = detector_feature_view_df.merge(
        resolved_rows_df.loc[:, resolved_cols],
        on="decision_row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(detector_feature_view_df):
        missing = sorted(
            set(detector_feature_view_df["decision_row_id"])
            - set(merged["decision_row_id"])
        )
        preview = missing[:5]
        raise ValueError(
            f"missing resolver rows for decision_row_id count={len(missing)} sample={preview}"
        )
    merged["trainable_row"] = merged["is_resolved"].astype(bool) & (
        ~merged["target_reason"]
        .astype(str)
        .str.lower()
        .isin(["invalid_context", "ambiguous"])
    )
    merged["detector_trainable_row"] = merged["trainable_row"].astype(int)
    ordered_columns = [
        *DETECTOR_IDENTITY_COLUMNS,
        *DETECTOR_FEATURE_COLUMNS,
        *TARGET_META_COLUMNS,
        "trainable_row",
        "detector_trainable_row",
    ]
    dataset = merged.loc[:, ordered_columns].copy()
    resolved_rows = int(dataset["is_resolved"].astype(bool).sum())
    trainable_rows = int(dataset["trainable_row"].sum())
    positive_rate = (
        float(
            dataset.loc[dataset["trainable_row"], "target_good_short_now"]
            .astype(float)
            .mean()
        )
        if trainable_rows > 0
        else 0.0
    )
    log_info(
        "DETECTOR",
        (
            f"dataset build done rows_total={len(dataset)} "
            f"resolved_rows={resolved_rows} trainable_rows={trainable_rows} positive_rate={positive_rate:.4f}"
        ),
    )
    stage_done("DETECTOR", "DATASET_BUILD", elapsed_sec=time.perf_counter() - started)
    return dataset


def _validate_unique_ids(df: pd.DataFrame, name: str) -> None:
    if "decision_row_id" not in df.columns:
        raise ValueError(f"{name} missing required column: decision_row_id")
    dup_mask = df["decision_row_id"].duplicated(keep=False)
    if dup_mask.any():
        duplicates = df.loc[dup_mask, "decision_row_id"].astype(str).head(5).tolist()
        raise ValueError(f"{name} has duplicate decision_row_id sample={duplicates}")


def _validate_feature_view_columns(df: pd.DataFrame) -> None:
    required = [*DETECTOR_IDENTITY_COLUMNS, *DETECTOR_FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"detector feature view missing columns: {missing}")
    leaked = [col for col in BLOCKED_COLUMNS if col in df.columns]
    if leaked:
        raise ValueError(f"detector feature view contains blocked columns: {leaked}")


def _validate_resolved_columns(df: pd.DataFrame) -> None:
    required = ["decision_row_id", *TARGET_META_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"resolved rows missing columns: {missing}")
