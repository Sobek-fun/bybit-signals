from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from pump_end_v2.config import DetectorCVConfig, ResolverConfig, SplitBounds
from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.logging import log_info


@dataclass(frozen=True, slots=True)
class DetectorFold:
    fold_id: str
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    purge_gap_bars: int
    train_row_count: int
    val_row_count: int


def assign_detector_dataset_splits(
    dataset_df: pd.DataFrame, split_bounds: SplitBounds
) -> pd.DataFrame:
    if "context_bar_open_time" not in dataset_df.columns:
        raise ValueError("dataset missing required column: context_bar_open_time")
    frame = dataset_df.copy()
    context_times = pd.to_datetime(
        frame["context_bar_open_time"], utc=True, errors="raise"
    )
    split = pd.Series("discard", index=frame.index, dtype="string")
    train_mask = context_times <= split_bounds.train_end
    val_mask = (context_times > split_bounds.train_end) & (
        context_times <= split_bounds.val_end
    )
    test_mask = (context_times > split_bounds.val_end) & (
        context_times <= split_bounds.test_end
    )
    split.loc[train_mask] = "train"
    split.loc[val_mask] = "val"
    split.loc[test_mask] = "test"
    frame["dataset_split"] = split
    summary = summarize_detector_splits(frame)
    log_info(
        "DETECTOR",
        (
            "split assignment summary "
            f"train_rows={summary['train_rows']} val_rows={summary['val_rows']} "
            f"test_rows={summary['test_rows']} discard_rows={summary['discard_rows']}"
        ),
    )
    return frame


def summarize_detector_splits(dataset_df: pd.DataFrame) -> dict[str, int]:
    if "dataset_split" not in dataset_df.columns:
        raise ValueError("dataset missing required column: dataset_split")
    counts = dataset_df["dataset_split"].value_counts(dropna=False).to_dict()
    return {
        "train_rows": int(counts.get("train", 0)),
        "val_rows": int(counts.get("val", 0)),
        "test_rows": int(counts.get("test", 0)),
        "discard_rows": int(counts.get("discard", 0)),
    }


def generate_detector_walkforward_folds(
    dataset_df: pd.DataFrame,
    split_bounds: SplitBounds,
    resolver_config: ResolverConfig,
    execution_contract: ExecutionContract,
    detector_cv_config: DetectorCVConfig,
) -> list[DetectorFold]:
    _validate_fold_inputs(dataset_df)
    train_df = dataset_df[dataset_df["dataset_split"] == "train"].copy()
    if train_df.empty:
        log_info(
            "CV",
            "fold generation summary folds_total=0 "
            f"purge_gap_bars={int(execution_contract.entry_shift_bars) + int(execution_contract.max_hold_bars)}",
        )
        return []
    train_df["context_bar_open_time"] = pd.to_datetime(
        train_df["context_bar_open_time"], utc=True, errors="raise"
    )
    train_df = train_df.sort_values(
        "context_bar_open_time", kind="mergesort"
    ).reset_index(drop=True)
    train_start = train_df["context_bar_open_time"].min()
    train_end = min(
        train_df["context_bar_open_time"].max(), pd.Timestamp(split_bounds.train_end)
    )
    min_train_delta = pd.Timedelta(days=detector_cv_config.min_train_days)
    fold_span_delta = pd.Timedelta(days=detector_cv_config.fold_span_days)
    purge_gap_bars = int(execution_contract.entry_shift_bars) + int(
        execution_contract.max_hold_bars
    )
    purge_gap_timedelta = pd.Timedelta(minutes=15 * purge_gap_bars)
    val_window_delta = fold_span_delta - pd.Timedelta(minutes=15)
    candidate_folds: list[DetectorFold] = []
    cursor = train_start + min_train_delta
    while cursor <= train_end:
        val_start = cursor
        val_end = min(val_start + val_window_delta, train_end)
        train_cutoff = val_start - purge_gap_timedelta
        fold_train = train_df[train_df["context_bar_open_time"] < train_cutoff]
        fold_val = train_df[
            (train_df["context_bar_open_time"] >= val_start)
            & (train_df["context_bar_open_time"] <= val_end)
        ]
        if not fold_train.empty and not fold_val.empty:
            candidate_folds.append(
                DetectorFold(
                    fold_id="",
                    train_start=fold_train["context_bar_open_time"]
                    .min()
                    .to_pydatetime(),
                    train_end=fold_train["context_bar_open_time"].max().to_pydatetime(),
                    val_start=val_start.to_pydatetime(),
                    val_end=val_end.to_pydatetime(),
                    purge_gap_bars=purge_gap_bars,
                    train_row_count=len(fold_train),
                    val_row_count=len(fold_val),
                )
            )
        cursor = cursor + fold_span_delta
    if len(candidate_folds) > detector_cv_config.max_folds:
        candidate_folds = candidate_folds[-detector_cv_config.max_folds :]
    folds: list[DetectorFold] = []
    for idx, fold in enumerate(candidate_folds, start=1):
        folds.append(
            DetectorFold(
                fold_id=f"fold_{idx}",
                train_start=fold.train_start,
                train_end=fold.train_end,
                val_start=fold.val_start,
                val_end=fold.val_end,
                purge_gap_bars=fold.purge_gap_bars,
                train_row_count=fold.train_row_count,
                val_row_count=fold.val_row_count,
            )
        )
    log_info(
        "CV",
        f"fold generation summary folds_total={len(folds)} purge_gap_bars={purge_gap_bars}",
    )
    return folds


def filter_fold_rows(
    dataset_df: pd.DataFrame, fold: DetectorFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_fold_inputs(dataset_df)
    train_split = dataset_df[dataset_df["dataset_split"] == "train"].copy()
    train_split["context_bar_open_time"] = pd.to_datetime(
        train_split["context_bar_open_time"], utc=True, errors="raise"
    )
    train_df = train_split[
        (train_split["context_bar_open_time"] >= fold.train_start)
        & (train_split["context_bar_open_time"] <= fold.train_end)
    ].copy()
    val_df = train_split[
        (train_split["context_bar_open_time"] >= fold.val_start)
        & (train_split["context_bar_open_time"] <= fold.val_end)
    ].copy()
    return train_df, val_df


def _validate_fold_inputs(dataset_df: pd.DataFrame) -> None:
    for col in ("dataset_split", "context_bar_open_time"):
        if col not in dataset_df.columns:
            raise ValueError(f"dataset missing required column: {col}")
