from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pump_end_v2.features.manifest import DETECTOR_FEATURE_COLUMNS


def build_detector_row_rank_report(policy_rows_df: pd.DataFrame) -> dict[str, float]:
    frame = _prepare_resolved_policy_rows(policy_rows_df)
    if frame.empty:
        return {
            "rows_total": 0.0,
            "resolved_rows_total": 0.0,
            "tp_rows_total": 0.0,
            "sl_rows_total": 0.0,
            "timeout_rows_total": 0.0,
            "p_good_nan_share": 0.0,
            "auc_tp_vs_non_tp": 0.0,
            "mean_p_good_tp": 0.0,
            "mean_p_good_sl": 0.0,
            "mean_p_good_timeout": 0.0,
            "median_p_good_tp": 0.0,
            "median_p_good_sl": 0.0,
            "median_p_good_timeout": 0.0,
            "corr_p_good_trade_pnl": 0.0,
            "corr_p_good_mae": 0.0,
            "top_decile_tp_rate": 0.0,
            "bottom_decile_tp_rate": 0.0,
            "top_decile_avg_trade_pnl_pct": 0.0,
            "bottom_decile_avg_trade_pnl_pct": 0.0,
            "top_decile_avg_mae_pct": 0.0,
            "bottom_decile_avg_mae_pct": 0.0,
        }
    p_good = pd.to_numeric(frame["p_good"], errors="coerce")
    tp_mask = frame["target_tp"].astype(bool)
    sl_mask = frame["outcome_label"].eq("sl")
    timeout_mask = frame["outcome_label"].eq("timeout")
    valid_auc = p_good.notna()
    auc = _binary_roc_auc(
        frame.loc[valid_auc, "target_tp"].astype(int),
        p_good.loc[valid_auc].astype(float),
    )
    top_stats = _row_decile_edge_stats(frame, edge="top")
    bottom_stats = _row_decile_edge_stats(frame, edge="bottom")
    return {
        "rows_total": float(frame["rows_total_source"].iloc[0]),
        "resolved_rows_total": float(len(frame)),
        "tp_rows_total": float(int(tp_mask.sum())),
        "sl_rows_total": float(int(sl_mask.sum())),
        "timeout_rows_total": float(int(timeout_mask.sum())),
        "p_good_nan_share": float(p_good.isna().mean()),
        "auc_tp_vs_non_tp": float(auc),
        "mean_p_good_tp": _safe_mean(p_good.loc[tp_mask]),
        "mean_p_good_sl": _safe_mean(p_good.loc[sl_mask]),
        "mean_p_good_timeout": _safe_mean(p_good.loc[timeout_mask]),
        "median_p_good_tp": _safe_median(p_good.loc[tp_mask]),
        "median_p_good_sl": _safe_median(p_good.loc[sl_mask]),
        "median_p_good_timeout": _safe_median(p_good.loc[timeout_mask]),
        "corr_p_good_trade_pnl": _safe_corr(p_good, frame["row_trade_pnl_pct"]),
        "corr_p_good_mae": _safe_corr(p_good, frame["row_mae_pct"]),
        "top_decile_tp_rate": float(top_stats["tp_rate"]),
        "bottom_decile_tp_rate": float(bottom_stats["tp_rate"]),
        "top_decile_avg_trade_pnl_pct": float(top_stats["avg_trade_pnl"]),
        "bottom_decile_avg_trade_pnl_pct": float(bottom_stats["avg_trade_pnl"]),
        "top_decile_avg_mae_pct": float(top_stats["avg_mae"]),
        "bottom_decile_avg_mae_pct": float(bottom_stats["avg_mae"]),
    }


def build_detector_row_decile_report(policy_rows_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decile",
        "rows_total",
        "tp_rows_total",
        "sl_rows_total",
        "timeout_rows_total",
        "tp_rate_resolved",
        "avg_p_good",
        "min_p_good",
        "max_p_good",
        "avg_trade_pnl_pct",
        "median_trade_pnl_pct",
        "avg_mae_pct",
        "median_mae_pct",
        "avg_episode_age_bars",
        "avg_distance_from_episode_high_pct",
    ]
    frame = _prepare_resolved_policy_rows(policy_rows_df)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    p_good = pd.to_numeric(frame["p_good"], errors="coerce")
    ranked = frame.loc[p_good.notna()].copy()
    ranked["p_good"] = p_good.loc[p_good.notna()].astype(float)
    if ranked.empty:
        return pd.DataFrame(columns=columns)
    deciles = _assign_deciles(ranked["p_good"])
    ranked["decile"] = deciles
    out_rows: list[dict[str, float | int]] = []
    for decile, group in ranked.groupby("decile", sort=True):
        tp_rows = int(group["target_tp"].astype(bool).sum())
        sl_rows = int(group["outcome_label"].eq("sl").sum())
        timeout_rows = int(group["outcome_label"].eq("timeout").sum())
        resolved = tp_rows + sl_rows + timeout_rows
        out_rows.append(
            {
                "decile": int(decile),
                "rows_total": int(len(group)),
                "tp_rows_total": tp_rows,
                "sl_rows_total": sl_rows,
                "timeout_rows_total": timeout_rows,
                "tp_rate_resolved": _safe_ratio(tp_rows, resolved),
                "avg_p_good": _safe_mean(group["p_good"]),
                "min_p_good": _safe_min(group["p_good"]),
                "max_p_good": _safe_max(group["p_good"]),
                "avg_trade_pnl_pct": _safe_mean(group["row_trade_pnl_pct"]),
                "median_trade_pnl_pct": _safe_median(group["row_trade_pnl_pct"]),
                "avg_mae_pct": _safe_mean(group["row_mae_pct"]),
                "median_mae_pct": _safe_median(group["row_mae_pct"]),
                "avg_episode_age_bars": _safe_mean(group["episode_age_bars"]),
                "avg_distance_from_episode_high_pct": _safe_mean(
                    group["distance_from_episode_high_pct"]
                ),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(
        "decile", kind="mergesort"
    ).reset_index(drop=True)


def build_detector_episode_rank_report(policy_rows_df: pd.DataFrame) -> dict[str, float]:
    episode_df = _build_episode_base_table(policy_rows_df)
    if episode_df.empty:
        return {
            "episodes_total": 0.0,
            "tradeable_episodes_total": 0.0,
            "nontradeable_episodes_total": 0.0,
            "auc_episode_max_p_good": 0.0,
            "auc_episode_mean_p_good": 0.0,
            "auc_episode_last_p_good": 0.0,
            "auc_episode_p_good_range": 0.0,
            "mean_max_p_good_tradeable": 0.0,
            "mean_max_p_good_nontradeable": 0.0,
            "mean_mean_p_good_tradeable": 0.0,
            "mean_mean_p_good_nontradeable": 0.0,
            "mean_p_good_range_tradeable": 0.0,
            "mean_p_good_range_nontradeable": 0.0,
            "median_first_tp_age_tradeable": 0.0,
            "mean_tp_row_count_tradeable": 0.0,
        }
    tradeable_mask = episode_df["tradeable_episode"].astype(bool)
    return {
        "episodes_total": float(len(episode_df)),
        "tradeable_episodes_total": float(int(tradeable_mask.sum())),
        "nontradeable_episodes_total": float(int((~tradeable_mask).sum())),
        "auc_episode_max_p_good": _binary_roc_auc(
            tradeable_mask.astype(int), episode_df["max_p_good"]
        ),
        "auc_episode_mean_p_good": _binary_roc_auc(
            tradeable_mask.astype(int), episode_df["mean_p_good"]
        ),
        "auc_episode_last_p_good": _binary_roc_auc(
            tradeable_mask.astype(int), episode_df["last_p_good"]
        ),
        "auc_episode_p_good_range": _binary_roc_auc(
            tradeable_mask.astype(int), episode_df["p_good_range"]
        ),
        "mean_max_p_good_tradeable": _safe_mean(
            episode_df.loc[tradeable_mask, "max_p_good"]
        ),
        "mean_max_p_good_nontradeable": _safe_mean(
            episode_df.loc[~tradeable_mask, "max_p_good"]
        ),
        "mean_mean_p_good_tradeable": _safe_mean(
            episode_df.loc[tradeable_mask, "mean_p_good"]
        ),
        "mean_mean_p_good_nontradeable": _safe_mean(
            episode_df.loc[~tradeable_mask, "mean_p_good"]
        ),
        "mean_p_good_range_tradeable": _safe_mean(
            episode_df.loc[tradeable_mask, "p_good_range"]
        ),
        "mean_p_good_range_nontradeable": _safe_mean(
            episode_df.loc[~tradeable_mask, "p_good_range"]
        ),
        "median_first_tp_age_tradeable": _safe_median(
            episode_df.loc[tradeable_mask, "first_tp_age"]
        ),
        "mean_tp_row_count_tradeable": _safe_mean(
            episode_df.loc[tradeable_mask, "tp_row_count"]
        ),
    }


def build_detector_episode_diagnostics(
    policy_rows_df: pd.DataFrame, candidate_signals_df: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "episode_id",
        "symbol",
        "rows_total",
        "tp_row_count",
        "tradeable_episode",
        "fired_episode",
        "episode_group",
        "first_tp_age",
        "max_p_good",
        "mean_p_good",
        "last_p_good",
        "p_good_range",
        "p_good_std",
        "max_p_good_age",
        "p_good_at_first_tp",
        "max_p_good_before_first_tp",
        "fire_age",
        "fire_p_good",
        "fire_before_first_tp",
        "fire_after_first_tp",
        "mean_mae_tp_rows",
        "mean_trade_pnl_tp_rows",
    ]
    episode_df = _build_episode_base_table(policy_rows_df)
    if episode_df.empty:
        return pd.DataFrame(columns=columns)
    fired_map = _build_fired_episode_map(candidate_signals_df)
    episode_df = episode_df.copy()
    episode_df["fired_episode"] = episode_df["episode_id"].astype(str).isin(fired_map)
    fire_info = episode_df["episode_id"].astype(str).map(fired_map)
    episode_df["fire_age"] = fire_info.map(
        lambda item: item[0]
        if isinstance(item, tuple) and len(item) >= 1
        else np.nan
    )
    episode_df["fire_p_good"] = fire_info.map(
        lambda item: item[1]
        if isinstance(item, tuple) and len(item) >= 2
        else np.nan
    )
    first_tp_age = pd.to_numeric(episode_df["first_tp_age"], errors="coerce")
    fire_age = pd.to_numeric(episode_df["fire_age"], errors="coerce")
    episode_df["fire_before_first_tp"] = (
        fire_age.notna() & first_tp_age.notna() & (fire_age <= first_tp_age)
    )
    episode_df["fire_after_first_tp"] = (
        fire_age.notna() & first_tp_age.notna() & (fire_age > first_tp_age)
    )
    episode_df["episode_group"] = episode_df.apply(_resolve_episode_group, axis=1)
    return episode_df.loc[:, columns].copy()


def build_detector_episode_group_summary(
    episode_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "episode_group",
        "episodes_total",
        "mean_first_tp_age",
        "median_first_tp_age",
        "mean_tp_row_count",
        "mean_max_p_good",
        "mean_mean_p_good",
        "mean_last_p_good",
        "mean_p_good_range",
        "mean_p_good_std",
        "mean_max_p_good_before_first_tp",
        "mean_p_good_at_first_tp",
        "mean_fire_age",
        "mean_fire_p_good",
        "share_fire_before_first_tp",
        "share_fire_after_first_tp",
        "mean_mae_tp_rows",
        "mean_trade_pnl_tp_rows",
    ]
    if episode_diagnostics_df.empty:
        return pd.DataFrame(columns=columns)
    frame = episode_diagnostics_df.copy()
    rows: list[dict[str, float | str]] = []
    for group_name, group in frame.groupby("episode_group", sort=True):
        rows.append(
            {
                "episode_group": str(group_name),
                "episodes_total": float(len(group)),
                "mean_first_tp_age": _safe_mean(group["first_tp_age"]),
                "median_first_tp_age": _safe_median(group["first_tp_age"]),
                "mean_tp_row_count": _safe_mean(group["tp_row_count"]),
                "mean_max_p_good": _safe_mean(group["max_p_good"]),
                "mean_mean_p_good": _safe_mean(group["mean_p_good"]),
                "mean_last_p_good": _safe_mean(group["last_p_good"]),
                "mean_p_good_range": _safe_mean(group["p_good_range"]),
                "mean_p_good_std": _safe_mean(group["p_good_std"]),
                "mean_max_p_good_before_first_tp": _safe_mean(
                    group["max_p_good_before_first_tp"]
                ),
                "mean_p_good_at_first_tp": _safe_mean(group["p_good_at_first_tp"]),
                "mean_fire_age": _safe_mean(group["fire_age"]),
                "mean_fire_p_good": _safe_mean(group["fire_p_good"]),
                "share_fire_before_first_tp": _safe_mean(group["fire_before_first_tp"]),
                "share_fire_after_first_tp": _safe_mean(group["fire_after_first_tp"]),
                "mean_mae_tp_rows": _safe_mean(group["mean_mae_tp_rows"]),
                "mean_trade_pnl_tp_rows": _safe_mean(group["mean_trade_pnl_tp_rows"]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        "episode_group", kind="mergesort"
    ).reset_index(drop=True)


def build_detector_feature_signal_report(
    detector_dataset_df: pd.DataFrame,
    resolved_rows_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    columns = [
        "feature",
        "split",
        "rows_total",
        "tp_rows_total",
        "non_tp_rows_total",
        "missing_share",
        "tp_mean",
        "non_tp_mean",
        "tp_median",
        "non_tp_median",
        "standardized_mean_diff",
        "univariate_auc_tp_vs_non_tp",
        "abs_auc_distance_from_0_5",
        "direction_tp_gt_non_tp",
    ]
    split = str(split_name).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split_name: {split_name}")
    if detector_dataset_df.empty:
        return pd.DataFrame(columns=columns)
    frame = detector_dataset_df.copy()
    if "dataset_split" not in frame.columns:
        raise ValueError("detector_dataset_df missing required column: dataset_split")
    frame = frame[frame["dataset_split"].astype(str).str.lower() == split].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "trainable_row" in frame.columns:
        frame = frame[frame["trainable_row"].astype(bool)].copy()
    if "is_resolved" in frame.columns:
        frame = frame[frame["is_resolved"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _ensure_truth_columns(frame, resolved_rows_df)
    truth = _derive_truth_target(frame)
    frame = frame.loc[truth["valid_truth"]].copy()
    frame["target_tp"] = truth.loc[truth["valid_truth"], "target_tp"].astype(int).to_numpy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    tp_total = int(frame["target_tp"].sum())
    non_tp_total = int((frame["target_tp"] == 0).sum())
    out_rows: list[dict[str, Any]] = []
    for feature in DETECTOR_FEATURE_COLUMNS:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        tp_values = values.loc[frame["target_tp"] == 1]
        non_tp_values = values.loc[frame["target_tp"] == 0]
        auc = _binary_roc_auc(frame["target_tp"], values)
        tp_mean = _safe_mean(tp_values)
        non_tp_mean = _safe_mean(non_tp_values)
        pooled_std = _safe_pooled_std(tp_values, non_tp_values)
        smd = (tp_mean - non_tp_mean) / pooled_std if pooled_std > 0.0 else 0.0
        out_rows.append(
            {
                "feature": str(feature),
                "split": split,
                "rows_total": float(len(frame)),
                "tp_rows_total": float(tp_total),
                "non_tp_rows_total": float(non_tp_total),
                "missing_share": float(values.isna().mean()),
                "tp_mean": float(tp_mean),
                "non_tp_mean": float(non_tp_mean),
                "tp_median": _safe_median(tp_values),
                "non_tp_median": _safe_median(non_tp_values),
                "standardized_mean_diff": float(smd),
                "univariate_auc_tp_vs_non_tp": float(auc),
                "abs_auc_distance_from_0_5": float(abs(auc - 0.5)),
                "direction_tp_gt_non_tp": bool(tp_mean > non_tp_mean),
            }
        )
    return pd.DataFrame(out_rows, columns=columns)


def build_detector_training_overfit_report(
    sequence_train_stats: dict[str, Any],
    training_history_df: pd.DataFrame,
) -> dict[str, float | int | bool | str]:
    history = training_history_df.copy()
    if history.empty:
        return {
            "monitor_name": str(sequence_train_stats.get("monitor_name", "")),
            "best_epoch": int(_safe_int(sequence_train_stats.get("best_epoch"), 0)),
            "epochs_ran": int(_safe_int(sequence_train_stats.get("epochs_ran"), 0)),
            "stopped_early": bool(sequence_train_stats.get("stopped_early", False)),
            "train_positive_rate": float(
                _safe_float(sequence_train_stats.get("train_positive_rate"), 0.0)
            ),
            "eval_positive_rate": float(
                _safe_float(sequence_train_stats.get("eval_positive_rate"), 0.0)
            ),
            "best_train_loss": 0.0,
            "best_eval_loss": 0.0,
            "final_train_loss": 0.0,
            "final_eval_loss": 0.0,
            "eval_loss_drift_from_best": 0.0,
            "train_loss_drift_from_best": 0.0,
            "best_epoch_fraction_of_training": 0.0,
            "overfit_started_epoch": 0,
            "best_classification_loss": float(
                _safe_float(sequence_train_stats.get("best_classification_loss"), 0.0)
            ),
            "best_ranking_loss": float(
                _safe_float(sequence_train_stats.get("best_ranking_loss"), 0.0)
            ),
            "final_classification_loss": float(
                _safe_float(sequence_train_stats.get("final_classification_loss"), 0.0)
            ),
            "final_ranking_loss": float(
                _safe_float(sequence_train_stats.get("final_ranking_loss"), 0.0)
            ),
            "ranking_pairs_train_total": int(
                _safe_int(sequence_train_stats.get("ranking_pairs_train_total"), 0)
            ),
            "ranking_pairs_eval_total": int(
                _safe_int(sequence_train_stats.get("ranking_pairs_eval_total"), 0)
            ),
            "hard_negative_rows_train_total": int(
                _safe_int(sequence_train_stats.get("hard_negative_rows_train_total"), 0)
            ),
            "hard_negative_rows_eval_total": int(
                _safe_int(sequence_train_stats.get("hard_negative_rows_eval_total"), 0)
            ),
        }
    history["epoch"] = pd.to_numeric(history.get("epoch"), errors="coerce")
    history["train_loss"] = pd.to_numeric(history.get("train_loss"), errors="coerce")
    history["eval_loss"] = pd.to_numeric(history.get("eval_loss"), errors="coerce")
    history = history.dropna(subset=["epoch"]).sort_values("epoch", kind="mergesort")
    if history.empty:
        return {
            "monitor_name": str(sequence_train_stats.get("monitor_name", "")),
            "best_epoch": int(_safe_int(sequence_train_stats.get("best_epoch"), 0)),
            "epochs_ran": int(_safe_int(sequence_train_stats.get("epochs_ran"), 0)),
            "stopped_early": bool(sequence_train_stats.get("stopped_early", False)),
            "train_positive_rate": float(
                _safe_float(sequence_train_stats.get("train_positive_rate"), 0.0)
            ),
            "eval_positive_rate": float(
                _safe_float(sequence_train_stats.get("eval_positive_rate"), 0.0)
            ),
            "best_train_loss": 0.0,
            "best_eval_loss": 0.0,
            "final_train_loss": 0.0,
            "final_eval_loss": 0.0,
            "eval_loss_drift_from_best": 0.0,
            "train_loss_drift_from_best": 0.0,
            "best_epoch_fraction_of_training": 0.0,
            "overfit_started_epoch": 0,
            "best_classification_loss": float(
                _safe_float(sequence_train_stats.get("best_classification_loss"), 0.0)
            ),
            "best_ranking_loss": float(
                _safe_float(sequence_train_stats.get("best_ranking_loss"), 0.0)
            ),
            "final_classification_loss": float(
                _safe_float(sequence_train_stats.get("final_classification_loss"), 0.0)
            ),
            "final_ranking_loss": float(
                _safe_float(sequence_train_stats.get("final_ranking_loss"), 0.0)
            ),
            "ranking_pairs_train_total": int(
                _safe_int(sequence_train_stats.get("ranking_pairs_train_total"), 0)
            ),
            "ranking_pairs_eval_total": int(
                _safe_int(sequence_train_stats.get("ranking_pairs_eval_total"), 0)
            ),
            "hard_negative_rows_train_total": int(
                _safe_int(sequence_train_stats.get("hard_negative_rows_train_total"), 0)
            ),
            "hard_negative_rows_eval_total": int(
                _safe_int(sequence_train_stats.get("hard_negative_rows_eval_total"), 0)
            ),
        }
    best_epoch_from_stats = _safe_int(sequence_train_stats.get("best_epoch"), 0)
    best_row = history.loc[history["epoch"] == float(best_epoch_from_stats)]
    if best_row.empty:
        best_row = history.loc[[history["eval_loss"].idxmin()]]
        best_epoch = int(_safe_int(best_row["epoch"].iloc[0], 0))
    else:
        best_epoch = int(best_epoch_from_stats)
    best_train_loss = _safe_float(best_row["train_loss"].iloc[0], 0.0)
    best_eval_loss = _safe_float(best_row["eval_loss"].iloc[0], 0.0)
    final_row = history.iloc[-1]
    final_train_loss = _safe_float(final_row["train_loss"], 0.0)
    final_eval_loss = _safe_float(final_row["eval_loss"], 0.0)
    epochs_ran = int(_safe_int(sequence_train_stats.get("epochs_ran"), len(history)))
    if epochs_ran <= 0:
        epochs_ran = int(len(history))
    overfit_started_epoch = 0
    after_best = history.loc[history["epoch"] > float(best_epoch)].copy()
    after_best = after_best.loc[after_best["eval_loss"] > float(best_eval_loss)]
    if not after_best.empty:
        overfit_started_epoch = int(_safe_int(after_best["epoch"].iloc[0], 0))
    return {
        "monitor_name": str(sequence_train_stats.get("monitor_name", "")),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "stopped_early": bool(sequence_train_stats.get("stopped_early", False)),
        "train_positive_rate": float(
            _safe_float(sequence_train_stats.get("train_positive_rate"), 0.0)
        ),
        "eval_positive_rate": float(
            _safe_float(sequence_train_stats.get("eval_positive_rate"), 0.0)
        ),
        "best_train_loss": float(best_train_loss),
        "best_eval_loss": float(best_eval_loss),
        "final_train_loss": float(final_train_loss),
        "final_eval_loss": float(final_eval_loss),
        "eval_loss_drift_from_best": float(final_eval_loss - best_eval_loss),
        "train_loss_drift_from_best": float(final_train_loss - best_train_loss),
        "best_epoch_fraction_of_training": float(
            float(best_epoch) / float(max(epochs_ran, 1))
        ),
        "overfit_started_epoch": int(overfit_started_epoch),
        "best_classification_loss": float(
            _safe_float(sequence_train_stats.get("best_classification_loss"), 0.0)
        ),
        "best_ranking_loss": float(
            _safe_float(sequence_train_stats.get("best_ranking_loss"), 0.0)
        ),
        "final_classification_loss": float(
            _safe_float(sequence_train_stats.get("final_classification_loss"), 0.0)
        ),
        "final_ranking_loss": float(
            _safe_float(sequence_train_stats.get("final_ranking_loss"), 0.0)
        ),
        "ranking_pairs_train_total": int(
            _safe_int(sequence_train_stats.get("ranking_pairs_train_total"), 0)
        ),
        "ranking_pairs_eval_total": int(
            _safe_int(sequence_train_stats.get("ranking_pairs_eval_total"), 0)
        ),
        "hard_negative_rows_train_total": int(
            _safe_int(sequence_train_stats.get("hard_negative_rows_train_total"), 0)
        ),
        "hard_negative_rows_eval_total": int(
            _safe_int(sequence_train_stats.get("hard_negative_rows_eval_total"), 0)
        ),
    }


def _prepare_resolved_policy_rows(policy_rows_df: pd.DataFrame) -> pd.DataFrame:
    frame = policy_rows_df.copy()
    if "p_good" not in frame.columns:
        raise ValueError("policy_rows_df missing required column: p_good")
    rows_total_source = float(len(frame))
    if "policy_context_only" in frame.columns:
        frame = frame[~frame["policy_context_only"].astype(bool)].copy()
    truth = _derive_truth_target(frame)
    frame = frame.loc[truth["valid_truth"]].copy()
    frame["target_tp"] = truth.loc[truth["valid_truth"], "target_tp"].astype(int).to_numpy()
    frame["outcome_label"] = truth.loc[truth["valid_truth"], "outcome_label"].astype(str).to_numpy()
    frame["p_good"] = pd.to_numeric(frame["p_good"], errors="coerce")
    frame["row_trade_pnl_pct"] = pd.to_numeric(
        frame.get("row_trade_pnl_pct", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    frame["row_mae_pct"] = pd.to_numeric(
        frame.get("row_mae_pct", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    frame["episode_age_bars"] = pd.to_numeric(
        frame.get("episode_age_bars", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    frame["distance_from_episode_high_pct"] = pd.to_numeric(
        frame.get(
            "distance_from_episode_high_pct", pd.Series(index=frame.index, dtype=float)
        ),
        errors="coerce",
    )
    frame["rows_total_source"] = rows_total_source
    return frame.reset_index(drop=True)


def _derive_truth_target(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if "row_trade_outcome" in frame.columns:
        outcome = (
            frame["row_trade_outcome"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan, "nan": np.nan, "none": np.nan})
        )
        valid = outcome.isin(["tp", "sl", "timeout"])
        out["valid_truth"] = valid
        out["target_tp"] = outcome.eq("tp").astype(int)
        out["outcome_label"] = outcome.where(valid, np.nan)
        return out
    if "target_good_short_now" not in frame.columns:
        raise ValueError(
            "policy_rows_df missing truth columns: row_trade_outcome and target_good_short_now"
        )
    target = pd.to_numeric(frame["target_good_short_now"], errors="coerce")
    valid = target.isin([0.0, 1.0])
    out["valid_truth"] = valid
    out["target_tp"] = (target == 1.0).astype(int)
    out["outcome_label"] = np.where(target == 1.0, "tp", "non_tp")
    return out


def _assign_deciles(values: pd.Series) -> pd.Series:
    count = int(len(values))
    if count <= 0:
        return pd.Series([], dtype=int, index=values.index)
    bins = min(10, count)
    ranks = values.rank(method="first")
    return (pd.qcut(ranks, q=bins, labels=False) + 1).astype(int)


def _row_decile_edge_stats(frame: pd.DataFrame, edge: str) -> dict[str, float]:
    ranked = frame.loc[frame["p_good"].notna()].copy()
    if ranked.empty:
        return {"tp_rate": 0.0, "avg_trade_pnl": 0.0, "avg_mae": 0.0}
    ranked["decile"] = _assign_deciles(ranked["p_good"])
    if edge == "top":
        selected = ranked.loc[ranked["decile"] == ranked["decile"].max()]
    else:
        selected = ranked.loc[ranked["decile"] == ranked["decile"].min()]
    if selected.empty:
        return {"tp_rate": 0.0, "avg_trade_pnl": 0.0, "avg_mae": 0.0}
    return {
        "tp_rate": _safe_mean(selected["target_tp"]),
        "avg_trade_pnl": _safe_mean(selected["row_trade_pnl_pct"]),
        "avg_mae": _safe_mean(selected["row_mae_pct"]),
    }


def _build_episode_base_table(policy_rows_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_resolved_policy_rows(policy_rows_df)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "symbol",
                "rows_total",
                "tp_row_count",
                "tradeable_episode",
                "first_tp_age",
                "max_p_good",
                "mean_p_good",
                "last_p_good",
                "p_good_range",
                "p_good_std",
                "max_p_good_age",
                "p_good_at_first_tp",
                "max_p_good_before_first_tp",
                "mean_mae_tp_rows",
                "mean_trade_pnl_tp_rows",
            ]
        )
    required = {"episode_id", "symbol"}
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"policy_rows_df missing required columns: {missing}")
    sort_cols = ["episode_id"]
    if "context_bar_open_time" in frame.columns:
        frame["context_bar_open_time"] = pd.to_datetime(
            frame["context_bar_open_time"], utc=True, errors="coerce"
        )
        sort_cols = ["episode_id", "context_bar_open_time"]
    elif "episode_age_bars" in frame.columns:
        sort_cols = ["episode_id", "episode_age_bars"]
    frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for episode_id, group in frame.groupby("episode_id", sort=False):
        g = group.copy()
        p_good = pd.to_numeric(g["p_good"], errors="coerce")
        age = pd.to_numeric(g["episode_age_bars"], errors="coerce")
        tp_mask = g["target_tp"].astype(bool)
        first_tp_age = _safe_min(age.loc[tp_mask]) if bool(tp_mask.any()) else 0.0
        if not bool(tp_mask.any()):
            first_tp_age = np.nan
        max_idx = p_good.idxmax() if p_good.notna().any() else None
        max_p_good_age = (
            _safe_float(age.loc[max_idx], 0.0) if max_idx is not None else 0.0
        )
        if max_idx is None:
            max_p_good_age = np.nan
        if bool(tp_mask.any()) and age.notna().any():
            min_tp_age = float(np.nanmin(age.loc[tp_mask].to_numpy(dtype=float)))
            before_tp = p_good.loc[age < min_tp_age]
            max_before_tp = _safe_max(before_tp)
            tp_at_first = p_good.loc[tp_mask & (age == min_tp_age)]
            p_good_at_first_tp = _safe_mean(tp_at_first)
        else:
            max_before_tp = 0.0
            p_good_at_first_tp = 0.0
            max_before_tp = np.nan
            p_good_at_first_tp = np.nan
        rows.append(
            {
                "episode_id": str(episode_id),
                "symbol": str(g["symbol"].iloc[0]),
                "rows_total": int(len(g)),
                "tp_row_count": int(tp_mask.sum()),
                "tradeable_episode": bool(tp_mask.any()),
                "first_tp_age": first_tp_age,
                "max_p_good": _safe_max(p_good),
                "mean_p_good": _safe_mean(p_good),
                "last_p_good": _safe_float(p_good.iloc[-1], 0.0),
                "p_good_range": _safe_range(p_good),
                "p_good_std": _safe_std(p_good),
                "max_p_good_age": max_p_good_age,
                "p_good_at_first_tp": p_good_at_first_tp,
                "max_p_good_before_first_tp": max_before_tp,
                "mean_mae_tp_rows": _safe_mean(g.loc[tp_mask, "row_mae_pct"]),
                "mean_trade_pnl_tp_rows": _safe_mean(g.loc[tp_mask, "row_trade_pnl_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _build_fired_episode_map(candidate_signals_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if candidate_signals_df is None or candidate_signals_df.empty:
        return {}
    if "episode_id" not in candidate_signals_df.columns:
        return {}
    frame = candidate_signals_df.copy()
    frame["episode_id"] = frame["episode_id"].astype(str)
    if "decision_time" in frame.columns:
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        frame = frame.sort_values("decision_time", kind="mergesort")
    elif "context_bar_open_time" in frame.columns:
        frame["context_bar_open_time"] = pd.to_datetime(
            frame["context_bar_open_time"], utc=True, errors="coerce"
        )
        frame = frame.sort_values("context_bar_open_time", kind="mergesort")
    frame = frame.drop_duplicates(subset=["episode_id"], keep="first")
    fire_age = pd.to_numeric(
        frame.get("episode_age_bars", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    fire_p_good = pd.to_numeric(
        frame.get("p_good", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    return {
        str(ep): (_safe_float(age, np.nan), _safe_float(score, np.nan))
        for ep, age, score in zip(frame["episode_id"], fire_age, fire_p_good)
    }


def _resolve_episode_group(row: pd.Series) -> str:
    tradeable = bool(row["tradeable_episode"])
    fired = bool(row["fired_episode"])
    if fired and tradeable:
        return "fired_tradeable"
    if (not fired) and tradeable:
        return "missed_tradeable"
    if fired and (not tradeable):
        return "fired_nontradeable"
    return "skipped_nontradeable"


def _ensure_truth_columns(
    frame: pd.DataFrame, resolved_rows_df: pd.DataFrame
) -> pd.DataFrame:
    required_truth = ["row_trade_outcome", "target_good_short_now", "is_resolved"]
    missing = [col for col in required_truth if col not in frame.columns]
    if not missing:
        return frame
    if "decision_row_id" not in frame.columns:
        return frame
    truth_cols = [
        col
        for col in ["decision_row_id", "row_trade_outcome", "target_good_short_now", "is_resolved"]
        if col in resolved_rows_df.columns
    ]
    if "decision_row_id" not in truth_cols:
        return frame
    merged = frame.merge(
        resolved_rows_df.loc[:, truth_cols],
        on="decision_row_id",
        how="left",
        suffixes=("", "_truth"),
        validate="one_to_one",
    )
    for col in ["row_trade_outcome", "target_good_short_now", "is_resolved"]:
        truth_col = f"{col}_truth"
        if col not in merged.columns and truth_col in merged.columns:
            merged[col] = merged[truth_col]
        elif truth_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[truth_col])
            merged = merged.drop(columns=[truth_col])
    return merged


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.mean())
    return 0.0 if not np.isfinite(out) else out


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.median())
    return 0.0 if not np.isfinite(out) else out


def _safe_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.std(ddof=0))
    return 0.0 if not np.isfinite(out) else out


def _safe_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.min())
    return 0.0 if not np.isfinite(out) else out


def _safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.max())
    return 0.0 if not np.isfinite(out) else out


def _safe_range(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    out = float(numeric.max() - numeric.min())
    return 0.0 if not np.isfinite(out) else out


def _safe_corr(values_a: pd.Series, values_b: pd.Series) -> float:
    a = pd.to_numeric(values_a, errors="coerce")
    b = pd.to_numeric(values_b, errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 2:
        return 0.0
    corr = a.loc[mask].corr(b.loc[mask])
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def _safe_pooled_std(values_a: pd.Series, values_b: pd.Series) -> float:
    a = pd.to_numeric(values_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(values_b, errors="coerce").dropna().to_numpy(dtype=float)
    if a.size == 0 and b.size == 0:
        return 0.0
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    n_a = float(max(a.size, 1))
    n_b = float(max(b.size, 1))
    pooled = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / max((n_a + n_b - 2.0), 1.0)
    if not np.isfinite(pooled) or pooled <= 0.0:
        return 0.0
    return float(math.sqrt(pooled))


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
