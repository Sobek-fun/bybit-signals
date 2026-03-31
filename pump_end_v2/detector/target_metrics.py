import pandas as pd


_PROBABILITY_COLUMNS: tuple[str, ...] = (
    "p_good",
    "p_too_early",
    "p_too_late",
    "p_continuation_flat",
)


def build_detector_target_metrics(policy_rows_df: pd.DataFrame) -> dict[str, float]:
    required_columns = (*_PROBABILITY_COLUMNS, "target_reason_group")
    missing = [
        column for column in required_columns if column not in policy_rows_df.columns
    ]
    if missing:
        raise ValueError(f"policy_rows_df missing required columns: {missing}")
    frame = policy_rows_df.copy()
    if "policy_context_only" in frame.columns:
        frame = frame[~frame["policy_context_only"].astype(bool)].copy()
    frame = frame[frame["target_reason_group"].notna()].copy()
    if frame.empty:
        return {
            "rows_total": 0.0,
            "multiclass_top1_accuracy": 0.0,
            "good_top1_recall": 0.0,
            "too_early_top1_recall": 0.0,
            "too_late_top1_recall": 0.0,
            "continuation_flat_top1_recall": 0.0,
            "predicted_good_on_too_early_rate": 0.0,
            "predicted_good_on_too_late_rate": 0.0,
            "predicted_good_on_continuation_flat_rate": 0.0,
            "mean_p_good_on_good": 0.0,
            "mean_p_too_early_on_too_early": 0.0,
            "mean_p_too_late_on_too_late": 0.0,
            "mean_p_continuation_flat_on_continuation_flat": 0.0,
        }
    for probability_column in _PROBABILITY_COLUMNS:
        frame[probability_column] = pd.to_numeric(
            frame[probability_column], errors="coerce"
        ).fillna(0.0)
    if "predicted_reason_group" in frame.columns:
        predicted_class = frame["predicted_reason_group"].astype(str).str.strip()
    else:
        predicted_class = (
            frame.loc[:, list(_PROBABILITY_COLUMNS)].idxmax(axis=1).str.removeprefix("p_")
        )
    target_class = frame["target_reason_group"].astype(str).str.strip()
    top1_correct = predicted_class == target_class
    predicted_good = predicted_class == "good"
    return {
        "rows_total": float(len(frame)),
        "multiclass_top1_accuracy": float(top1_correct.mean()),
        "good_top1_recall": _class_recall(top1_correct, target_class, "good"),
        "too_early_top1_recall": _class_recall(top1_correct, target_class, "too_early"),
        "too_late_top1_recall": _class_recall(top1_correct, target_class, "too_late"),
        "continuation_flat_top1_recall": _class_recall(
            top1_correct, target_class, "continuation_flat"
        ),
        "predicted_good_on_too_early_rate": _class_mask_rate(
            predicted_good, target_class, "too_early"
        ),
        "predicted_good_on_too_late_rate": _class_mask_rate(
            predicted_good, target_class, "too_late"
        ),
        "predicted_good_on_continuation_flat_rate": _class_mask_rate(
            predicted_good, target_class, "continuation_flat"
        ),
        "mean_p_good_on_good": _class_probability_mean(frame, "p_good", target_class, "good"),
        "mean_p_too_early_on_too_early": _class_probability_mean(
            frame, "p_too_early", target_class, "too_early"
        ),
        "mean_p_too_late_on_too_late": _class_probability_mean(
            frame, "p_too_late", target_class, "too_late"
        ),
        "mean_p_continuation_flat_on_continuation_flat": _class_probability_mean(
            frame, "p_continuation_flat", target_class, "continuation_flat"
        ),
    }


def _class_recall(
        top1_correct: pd.Series, target_class: pd.Series, label: str
) -> float:
    label_mask = target_class == label
    return _safe_ratio(int((top1_correct & label_mask).sum()), int(label_mask.sum()))


def _class_mask_rate(
        predicted_mask: pd.Series, target_class: pd.Series, label: str
) -> float:
    label_mask = target_class == label
    return _safe_ratio(int((predicted_mask & label_mask).sum()), int(label_mask.sum()))


def _class_probability_mean(
        frame: pd.DataFrame, probability_column: str, target_class: pd.Series, label: str
) -> float:
    label_mask = target_class == label
    if int(label_mask.sum()) <= 0:
        return 0.0
    return float(
        pd.to_numeric(frame.loc[label_mask, probability_column], errors="coerce")
        .fillna(0.0)
        .mean()
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)
