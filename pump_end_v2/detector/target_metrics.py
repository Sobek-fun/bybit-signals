import numpy as np
import pandas as pd


def build_detector_target_metrics(
    policy_rows_df: pd.DataFrame, good_threshold: float = 0.5
) -> dict[str, float]:
    required_columns = ("p_good", "target_reason")
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
            "tp_row_precision": 0.0,
            "tp_row_recall": 0.0,
            "sl_fp_rate": 0.0,
            "timeout_fp_rate": 0.0,
            "ambiguous_fp_rate": 0.0,
            "roc_auc_tp_vs_non_tp": 0.0,
            "mean_p_good_tp": 0.0,
            "mean_p_good_sl": 0.0,
            "mean_p_good_timeout": 0.0,
            "mean_p_good_ambiguous": 0.0,
        }
    p_good_numeric = pd.to_numeric(frame["p_good"], errors="coerce")
    p_good_nan_share = float(p_good_numeric.isna().mean())
    predicted_good = p_good_numeric.fillna(0.0) >= float(good_threshold)
    reasons = frame["target_reason"].astype(str).str.strip().str.lower()
    actual_good = reasons.eq("tp")
    valid_auc_mask = p_good_numeric.notna()
    auc_score = _binary_roc_auc(
        actual_good.loc[valid_auc_mask].astype(int),
        p_good_numeric.loc[valid_auc_mask].astype(float),
    )
    tp = int((predicted_good & actual_good).sum())
    pred_pos = int(predicted_good.sum())
    actual_pos = int(actual_good.sum())
    return {
        "rows_total": float(len(frame)),
        "p_good_nan_share": p_good_nan_share,
        "tp_row_precision": _safe_ratio(tp, pred_pos),
        "tp_row_recall": _safe_ratio(tp, actual_pos),
        "sl_fp_rate": _reason_fp_rate(predicted_good, reasons, "sl"),
        "timeout_fp_rate": _reason_fp_rate(predicted_good, reasons, "timeout"),
        "ambiguous_fp_rate": _reason_fp_rate(predicted_good, reasons, "ambiguous"),
        "roc_auc_tp_vs_non_tp": float(auc_score),
        "mean_p_good_tp": _mean_p_good_for_reason(p_good_numeric, reasons, "tp"),
        "mean_p_good_sl": _mean_p_good_for_reason(p_good_numeric, reasons, "sl"),
        "mean_p_good_timeout": _mean_p_good_for_reason(p_good_numeric, reasons, "timeout"),
        "mean_p_good_ambiguous": _mean_p_good_for_reason(
            p_good_numeric, reasons, "ambiguous"
        ),
    }


def build_detector_score_decile_report(policy_rows_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = (
        "p_good",
        "target_reason",
        "row_trade_pnl_pct",
        "row_mfe_pct",
        "row_mae_pct",
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
                "tp_rate",
                "sl_share",
                "timeout_share",
                "ambiguous_share",
                "avg_row_trade_pnl_pct",
                "mean_row_mfe_pct",
                "mean_row_mae_pct",
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
                "tp_rate",
                "sl_share",
                "timeout_share",
                "ambiguous_share",
                "avg_row_trade_pnl_pct",
                "mean_row_mfe_pct",
                "mean_row_mae_pct",
            ]
        )
    quantiles = min(10, int(len(frame)))
    ranks = frame["p_good"].rank(method="first")
    frame["decile"] = pd.qcut(ranks, q=quantiles, labels=False) + 1
    frame["target_reason"] = frame["target_reason"].astype(str).str.strip().str.lower()
    frame["row_trade_pnl_pct"] = pd.to_numeric(frame["row_trade_pnl_pct"], errors="coerce")
    frame["row_mfe_pct"] = pd.to_numeric(frame["row_mfe_pct"], errors="coerce")
    frame["row_mae_pct"] = pd.to_numeric(frame["row_mae_pct"], errors="coerce")
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
                "tp_rate": _safe_ratio(int((reasons == "tp").sum()), rows_total),
                "sl_share": _safe_ratio(int((reasons == "sl").sum()), rows_total),
                "timeout_share": _safe_ratio(int((reasons == "timeout").sum()), rows_total),
                "ambiguous_share": _safe_ratio(int((reasons == "ambiguous").sum()), rows_total),
                "avg_row_trade_pnl_pct": float(group["row_trade_pnl_pct"].mean()),
                "mean_row_mfe_pct": float(group["row_mfe_pct"].mean()),
                "mean_row_mae_pct": float(group["row_mae_pct"].mean()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("decile", kind="mergesort").reset_index(drop=True)


def build_detector_rank_quality_report(policy_rows_df: pd.DataFrame) -> dict[str, float]:
    required_columns = ("p_good", "target_reason")
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
            "roc_auc_tp_vs_non_tp": 0.0,
            "mean_p_good_tp": 0.0,
            "mean_p_good_non_tp": 0.0,
            "top_decile_tp_rate": 0.0,
            "bottom_decile_tp_rate": 0.0,
        }
    frame["p_good"] = pd.to_numeric(frame["p_good"], errors="coerce")
    frame["target_reason"] = frame["target_reason"].astype(str).str.strip().str.lower()
    frame["target_tp"] = frame["target_reason"].eq("tp").astype(int)
    frame = frame[frame["p_good"].notna()].copy()
    if frame.empty:
        return {
            "rows_total": 0.0,
            "roc_auc_tp_vs_non_tp": 0.0,
            "mean_p_good_tp": 0.0,
            "mean_p_good_non_tp": 0.0,
            "top_decile_tp_rate": 0.0,
            "bottom_decile_tp_rate": 0.0,
        }
    auc_score = _binary_roc_auc(frame["target_tp"], frame["p_good"])
    mean_p_good_tp = float(frame.loc[frame["target_tp"] == 1, "p_good"].mean())
    mean_p_good_non_tp = float(frame.loc[frame["target_tp"] != 1, "p_good"].mean())
    quantiles = min(10, int(len(frame)))
    ranks = frame["p_good"].rank(method="first")
    frame["decile"] = pd.qcut(ranks, q=quantiles, labels=False) + 1
    top_decile = int(frame["decile"].max())
    bottom_decile = int(frame["decile"].min())
    top_decile_tp_rate = float(
        frame.loc[frame["decile"] == top_decile, "target_tp"].mean()
    )
    bottom_decile_tp_rate = float(
        frame.loc[frame["decile"] == bottom_decile, "target_tp"].mean()
    )
    return {
        "rows_total": float(len(frame)),
        "roc_auc_tp_vs_non_tp": float(auc_score),
        "mean_p_good_tp": 0.0 if np.isnan(mean_p_good_tp) else float(mean_p_good_tp),
        "mean_p_good_non_tp": 0.0
        if np.isnan(mean_p_good_non_tp)
        else float(mean_p_good_non_tp),
        "top_decile_tp_rate": float(top_decile_tp_rate),
        "bottom_decile_tp_rate": float(bottom_decile_tp_rate),
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
