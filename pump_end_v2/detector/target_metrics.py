import pandas as pd


def build_detector_target_metrics(
        policy_rows_df: pd.DataFrame, good_threshold: float = 0.5
) -> dict[str, float]:
    required_columns = ("p_good", "target_good_short_now", "target_reason")
    missing = [
        column for column in required_columns if column not in policy_rows_df.columns
    ]
    if missing:
        raise ValueError(f"policy_rows_df missing required columns: {missing}")
    frame = policy_rows_df.copy()
    if "policy_context_only" in frame.columns:
        frame = frame[~frame["policy_context_only"].astype(bool)].copy()
    if frame.empty:
        return {
            "rows_total": 0.0,
            "p_good_nan_share": 0.0,
            "good_row_precision": 0.0,
            "good_row_recall": 0.0,
            "too_early_fp_rate": 0.0,
            "too_late_fp_rate": 0.0,
            "continuation_fp_rate": 0.0,
            "flat_fp_rate": 0.0,
        }
    p_good_numeric = pd.to_numeric(frame["p_good"], errors="coerce")
    p_good_nan_share = float(p_good_numeric.isna().mean())
    predicted_good = p_good_numeric.fillna(0.0) >= float(good_threshold)
    actual_good = (
            pd.to_numeric(frame["target_good_short_now"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
    )
    tp = int((predicted_good & actual_good).sum())
    pred_pos = int(predicted_good.sum())
    actual_pos = int(actual_good.sum())
    reasons = frame["target_reason"].astype(str).str.strip().str.lower()
    return {
        "rows_total": float(len(frame)),
        "p_good_nan_share": p_good_nan_share,
        "good_row_precision": _safe_ratio(tp, pred_pos),
        "good_row_recall": _safe_ratio(tp, actual_pos),
        "too_early_fp_rate": _reason_fp_rate(predicted_good, reasons, "too_early"),
        "too_late_fp_rate": _reason_fp_rate(predicted_good, reasons, "too_late"),
        "continuation_fp_rate": _reason_fp_rate(
            predicted_good, reasons, "continuation"
        ),
        "flat_fp_rate": _reason_fp_rate(predicted_good, reasons, "flat"),
    }


def _reason_fp_rate(
        predicted_good: pd.Series, reasons: pd.Series, reason_label: str
) -> float:
    mask = reasons == reason_label
    return _safe_ratio(int((predicted_good & mask).sum()), int(mask.sum()))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)
