import numpy as np
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
            "roc_auc_good_vs_bad": 0.0,
            "mean_p_good_good": 0.0,
            "mean_p_good_too_early": 0.0,
            "mean_p_good_too_late": 0.0,
            "mean_p_good_continuation": 0.0,
            "mean_p_good_flat": 0.0,
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
    valid_auc_mask = p_good_numeric.notna()
    auc_score = _binary_roc_auc(
        actual_good.loc[valid_auc_mask].astype(int),
        p_good_numeric.loc[valid_auc_mask].astype(float),
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
        "roc_auc_good_vs_bad": float(auc_score),
        "mean_p_good_good": _mean_p_good_for_reason(p_good_numeric, reasons, "good"),
        "mean_p_good_too_early": _mean_p_good_for_reason(p_good_numeric, reasons, "too_early"),
        "mean_p_good_too_late": _mean_p_good_for_reason(p_good_numeric, reasons, "too_late"),
        "mean_p_good_continuation": _mean_p_good_for_reason(
            p_good_numeric, reasons, "continuation"
        ),
        "mean_p_good_flat": _mean_p_good_for_reason(p_good_numeric, reasons, "flat"),
    }


def build_detector_score_decile_report(policy_rows_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = (
        "p_good",
        "target_reason",
        "target_good_short_now",
        "future_net_edge_pct",
        "future_pullback_pct",
        "future_prepullback_squeeze_pct",
    )
    missing = [
        column for column in required_columns if column not in policy_rows_df.columns
    ]
    if missing:
        raise ValueError(f"policy_rows_df missing required columns: {missing}")
    frame = policy_rows_df.copy()
    if "policy_context_only" in frame.columns:
        frame = frame[~frame["policy_context_only"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "decile",
                "rows",
                "p_good_min",
                "p_good_max",
                "p_good_mean",
                "good_rate",
                "too_early_share",
                "too_late_share",
                "continuation_share",
                "flat_share",
                "avg_future_net_edge_pct",
                "mean_future_pullback_pct",
                "mean_future_prepullback_squeeze_pct",
            ]
        )
    frame["p_good"] = pd.to_numeric(frame["p_good"], errors="coerce")
    frame = frame[frame["p_good"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "decile",
                "rows",
                "p_good_min",
                "p_good_max",
                "p_good_mean",
                "good_rate",
                "too_early_share",
                "too_late_share",
                "continuation_share",
                "flat_share",
                "avg_future_net_edge_pct",
                "mean_future_pullback_pct",
                "mean_future_prepullback_squeeze_pct",
            ]
        )
    quantiles = min(10, int(len(frame)))
    ranks = frame["p_good"].rank(method="first")
    frame["decile"] = pd.qcut(ranks, q=quantiles, labels=False) + 1
    frame["target_reason"] = frame["target_reason"].astype(str).str.strip().str.lower()
    frame["target_good_short_now"] = (
        pd.to_numeric(frame["target_good_short_now"], errors="coerce").fillna(0.0).astype(float)
    )
    frame["future_net_edge_pct"] = pd.to_numeric(
        frame["future_net_edge_pct"], errors="coerce"
    )
    frame["future_pullback_pct"] = pd.to_numeric(
        frame["future_pullback_pct"], errors="coerce"
    )
    frame["future_prepullback_squeeze_pct"] = pd.to_numeric(
        frame["future_prepullback_squeeze_pct"], errors="coerce"
    )
    out_rows: list[dict[str, float | int]] = []
    for decile_value, group in frame.groupby("decile", sort=True):
        rows_total = int(len(group))
        reasons = group["target_reason"]
        out_rows.append(
            {
                "decile": int(decile_value),
                "rows": rows_total,
                "p_good_min": float(group["p_good"].min()),
                "p_good_max": float(group["p_good"].max()),
                "p_good_mean": float(group["p_good"].mean()),
                "good_rate": float(group["target_good_short_now"].mean()),
                "too_early_share": _safe_ratio(int((reasons == "too_early").sum()), rows_total),
                "too_late_share": _safe_ratio(int((reasons == "too_late").sum()), rows_total),
                "continuation_share": _safe_ratio(
                    int((reasons == "continuation").sum()), rows_total
                ),
                "flat_share": _safe_ratio(int((reasons == "flat").sum()), rows_total),
                "avg_future_net_edge_pct": float(group["future_net_edge_pct"].mean()),
                "mean_future_pullback_pct": float(group["future_pullback_pct"].mean()),
                "mean_future_prepullback_squeeze_pct": float(
                    group["future_prepullback_squeeze_pct"].mean()
                ),
            }
        )
    return pd.DataFrame(out_rows).sort_values("decile", kind="mergesort").reset_index(drop=True)


def build_detector_rank_quality_report(policy_rows_df: pd.DataFrame) -> dict[str, float]:
    required_columns = ("p_good", "target_good_short_now")
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
            "roc_auc_good_vs_bad": 0.0,
            "mean_p_good_good": 0.0,
            "mean_p_good_bad": 0.0,
            "top_decile_good_rate": 0.0,
            "bottom_decile_good_rate": 0.0,
        }
    frame["p_good"] = pd.to_numeric(frame["p_good"], errors="coerce")
    frame["target_good_short_now"] = (
        pd.to_numeric(frame["target_good_short_now"], errors="coerce").fillna(0.0).astype(int)
    )
    frame = frame[frame["p_good"].notna()].copy()
    if frame.empty:
        return {
            "rows_total": 0.0,
            "roc_auc_good_vs_bad": 0.0,
            "mean_p_good_good": 0.0,
            "mean_p_good_bad": 0.0,
            "top_decile_good_rate": 0.0,
            "bottom_decile_good_rate": 0.0,
        }
    auc_score = _binary_roc_auc(frame["target_good_short_now"], frame["p_good"])
    mean_p_good_good = float(frame.loc[frame["target_good_short_now"] == 1, "p_good"].mean())
    mean_p_good_bad = float(frame.loc[frame["target_good_short_now"] != 1, "p_good"].mean())
    quantiles = min(10, int(len(frame)))
    ranks = frame["p_good"].rank(method="first")
    frame["decile"] = pd.qcut(ranks, q=quantiles, labels=False) + 1
    top_decile = int(frame["decile"].max())
    bottom_decile = int(frame["decile"].min())
    top_decile_good_rate = float(
        frame.loc[frame["decile"] == top_decile, "target_good_short_now"].mean()
    )
    bottom_decile_good_rate = float(
        frame.loc[frame["decile"] == bottom_decile, "target_good_short_now"].mean()
    )
    return {
        "rows_total": float(len(frame)),
        "roc_auc_good_vs_bad": float(auc_score),
        "mean_p_good_good": 0.0 if np.isnan(mean_p_good_good) else float(mean_p_good_good),
        "mean_p_good_bad": 0.0 if np.isnan(mean_p_good_bad) else float(mean_p_good_bad),
        "top_decile_good_rate": float(top_decile_good_rate),
        "bottom_decile_good_rate": float(bottom_decile_good_rate),
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


def _mean_p_good_for_reason(
        p_good_numeric: pd.Series, reasons: pd.Series, reason_label: str
) -> float:
    values = p_good_numeric.loc[reasons == reason_label]
    if len(values) == 0:
        return 0.0
    mean_value = float(values.mean())
    if np.isnan(mean_value):
        return 0.0
    return mean_value


def _binary_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    y_true_np = pd.to_numeric(y_true, errors="coerce").fillna(0.0).astype(int).to_numpy()
    y_score_np = pd.to_numeric(y_score, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(y_score_np)
    y_true_np = y_true_np[valid_mask]
    y_score_np = y_score_np[valid_mask]
    if y_true_np.size == 0:
        return 0.0
    positives = int(np.sum(y_true_np == 1))
    negatives = int(np.sum(y_true_np != 1))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(y_score_np)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score_np) + 1, dtype=float)
    _, inverse, counts = np.unique(y_score_np, return_inverse=True, return_counts=True)
    for value_idx, count in enumerate(counts):
        if count <= 1:
            continue
        idxs = np.where(inverse == value_idx)[0]
        avg_rank = float(np.mean(ranks[idxs]))
        ranks[idxs] = avg_rank
    rank_sum_pos = float(np.sum(ranks[y_true_np == 1]))
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / float(positives * negatives)
    if not np.isfinite(auc):
        return 0.0
    return float(max(0.0, min(1.0, auc)))
