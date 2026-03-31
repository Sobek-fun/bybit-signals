import pandas as pd


def build_detector_target_metrics(
    policy_rows_df: pd.DataFrame,
    target_column: str = "target_good_short_now",
    reason_column: str | None = "target_reason",
    good_threshold: float = 0.5,
) -> dict[str, float]:
    required_columns = ["p_good", target_column]
    if reason_column is not None:
        required_columns.append(reason_column)
    missing = [
        column for column in required_columns if column not in policy_rows_df.columns
    ]
    if missing:
        raise ValueError(f"policy_rows_df missing required columns: {missing}")
    frame = policy_rows_df.copy()
    if "policy_context_only" in frame.columns:
        frame = frame[~frame["policy_context_only"].astype(bool)].copy()
    if frame.empty:
        base = {
            "rows_total": 0.0,
            "good_row_precision": 0.0,
            "good_row_recall": 0.0,
        }
        if reason_column is not None:
            base.update(
                {
                    "too_early_fp_rate": 0.0,
                    "too_late_fp_rate": 0.0,
                    "continuation_fp_rate": 0.0,
                    "flat_fp_rate": 0.0,
                }
            )
        return base
    predicted_good = pd.to_numeric(frame["p_good"], errors="coerce").fillna(
        0.0
    ) >= float(good_threshold)
    actual_good = (
        pd.to_numeric(frame[target_column], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    tp = int((predicted_good & actual_good).sum())
    pred_pos = int(predicted_good.sum())
    actual_pos = int(actual_good.sum())
    metrics = {
        "rows_total": float(len(frame)),
        "good_row_precision": _safe_ratio(tp, pred_pos),
        "good_row_recall": _safe_ratio(tp, actual_pos),
    }
    if reason_column is not None:
        reasons = frame[reason_column].astype(str).str.strip().str.lower()
        metrics.update(
            {
                "too_early_fp_rate": _reason_fp_rate(
                    predicted_good, reasons, "too_early"
                ),
                "too_late_fp_rate": _reason_fp_rate(
                    predicted_good, reasons, "too_late"
                ),
                "continuation_fp_rate": _reason_fp_rate(
                    predicted_good, reasons, "continuation"
                ),
                "flat_fp_rate": _reason_fp_rate(predicted_good, reasons, "flat"),
            }
        )
    return metrics


def _reason_fp_rate(
    predicted_good: pd.Series, reasons: pd.Series, reason_label: str
) -> float:
    mask = reasons == reason_label
    return _safe_ratio(int((predicted_good & mask).sum()), int(mask.sum()))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)
