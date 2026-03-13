from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_FILES = [
    "run_config.json",
    "dataset_manifest.json",
    "regime_builder_config.json",
    "liquid_universe.json",
    "best_model_params.json",
    "best_policy_params.json",
    "policy_report.json",
    "cv_report.json",
    "fold_metrics.csv",
    "feature_columns.json",
    "feature_importance.csv",
    "feature_importance_grouped.csv",
    "test_scorecard.json",
    "test_metrics_full.json",
    "test_metrics.json",
    "test_scored.parquet",
    "test_predictions.parquet",
    "monthly_backtest.csv",
    "pause_episodes.csv",
    "p_bad_deciles.csv",
    "model_leaderboard.csv",
    "policy_leaderboard.csv",
    "leaderboard.csv",
    "guard_scored_oos.parquet",
    "guard_scored_signals.parquet",
    "blocked_oos.parquet",
    "accepted_oos.parquet",
    "blocked_signals.parquet",
    "accepted_signals.parquet",
    "raw_detector_signals_oos.parquet",
    "raw_detector_signals.parquet",
    "final_guarded_signals_oos.csv",
    "final_signals.csv",
    "regime_dataset.parquet",
]


def first_existing(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_table(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.{digits}f}".replace(",", " ")
        return f"{float(value):.{digits}f}"
    return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    return f"{float(value) * 100:.{digits}f}%"


def esc(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(esc(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def feature_group(name: str) -> str:
    if name.startswith("det_"):
        return "det"
    if name.startswith("snap_"):
        return "snap"
    if name.startswith("token_vs_") or name.startswith(
            ("token_relative_", "token_overheated_", "token_in_top_", "token_vol_spike_relative",
             "token_breakout_vs_")):
        return "token_vs"
    if name.startswith("token_"):
        return "token"
    if name.startswith("btc_eth_") or name.startswith(
            ("btc_strong_", "btc_weak_", "both_strong", "both_weak", "extreme_")):
        return "btc_eth"
    if name.startswith("btc_"):
        return "btc"
    if name.startswith("eth_"):
        return "eth"
    if name.startswith("breadth_"):
        return "breadth"
    if name.startswith("raw_signals_"):
        return "raw_signals"
    if name.startswith("unique_symbols_") or name.startswith("max_symbol_"):
        return "unique_symbols"
    if name.startswith("same_symbol_"):
        return "same_symbol"
    if name.startswith("signal_density_"):
        return "signal_density"
    if name.startswith("bucket_"):
        return "bucket"
    if name.startswith("strat_") or name.startswith(("resolved_", "open_trades_", "prev_closed_")):
        return "strat"
    return "other"


def summarize_cv(fold_metrics: pd.DataFrame, cv_report: dict[str, Any] | None) -> tuple[dict[str, Any], pd.DataFrame]:
    if fold_metrics.empty and cv_report and cv_report.get("fold_results"):
        fold_metrics = pd.DataFrame(cv_report["fold_results"])
    if fold_metrics.empty:
        return {}, fold_metrics
    valid = fold_metrics[fold_metrics.get("valid", True) == True].copy()
    total = len(fold_metrics)
    valid_count = len(valid)
    no_op = int(valid.get("no_op", pd.Series(dtype=bool)).fillna(False).sum()) if valid_count else 0
    positive = int((valid.get("pnl_improvement", pd.Series(dtype=float)).fillna(0) > 0).sum()) if valid_count else 0
    negative = int((valid.get("pnl_improvement", pd.Series(dtype=float)).fillna(0) < 0).sum()) if valid_count else 0
    summary = {
        "folds_total": total,
        "folds_valid": valid_count,
        "folds_no_op": no_op,
        "folds_positive_pnl": positive,
        "folds_negative_pnl": negative,
        "mean_score": valid["score"].mean() if "score" in valid else np.nan,
        "std_score": valid["score"].std(ddof=0) if "score" in valid else np.nan,
        "median_score": valid["score"].median() if "score" in valid else np.nan,
        "mean_pnl_improvement": valid["pnl_improvement"].mean() if "pnl_improvement" in valid else np.nan,
        "median_pnl_improvement": valid["pnl_improvement"].median() if "pnl_improvement" in valid else np.nan,
        "mean_blocked_share": valid["blocked_share"].mean() if "blocked_share" in valid else np.nan,
        "mean_signal_keep_rate": valid["signal_keep_rate"].mean() if "signal_keep_rate" in valid else np.nan,
        "mean_blocked_bad_precision": valid[
            "blocked_bad_precision"].mean() if "blocked_bad_precision" in valid else np.nan,
        "mean_sl_capture": valid["sl_capture"].mean() if "sl_capture" in valid else np.nan,
        "mean_tp_tax": valid["tp_tax"].mean() if "tp_tax" in valid else np.nan,
        "mean_episode_recall": valid["episode_recall"].mean() if "episode_recall" in valid else np.nan,
        "mean_worst_window_improvement_12h": valid[
            "worst_window_improvement_12h"].mean() if "worst_window_improvement_12h" in valid else np.nan,
        "mean_brier_score": valid["brier_score"].mean() if "brier_score" in valid else np.nan,
        "mean_p_bad_p90": valid["p_bad_p90"].mean() if "p_bad_p90" in valid else np.nan,
        "mean_p_bad_p95": valid["p_bad_p95"].mean() if "p_bad_p95" in valid else np.nan,
        "mean_p_bad_p99": valid["p_bad_p99"].mean() if "p_bad_p99" in valid else np.nan,
    }
    return summary, valid


def summarize_dataset(dataset: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    if dataset.empty:
        return {}
    summary = {
        "rows": len(dataset),
        "date_from": pd.to_datetime(dataset["open_time"]).min() if "open_time" in dataset else None,
        "date_to": pd.to_datetime(dataset["open_time"]).max() if "open_time" in dataset else None,
        "symbols": dataset["symbol"].nunique() if "symbol" in dataset else None,
        "target_rate": dataset[target_col].dropna().mean() if target_col and target_col in dataset else None,
        "target_valid_rows": dataset[target_col].notna().sum() if target_col and target_col in dataset else None,
    }
    return summary


def summarize_importance(fi: pd.DataFrame, grouped: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if fi.empty:
        return {}, fi, grouped
    fi = fi.copy()
    fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce").fillna(0.0)
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    if grouped.empty:
        tmp = fi.copy()
        tmp["feature_group"] = tmp["feature"].astype(str).map(feature_group)
        grouped = (
            tmp.groupby("feature_group", as_index=False)
            .agg(sum_importance=("importance", "sum"), mean_importance=("importance", "mean"),
                 count=("feature", "count"))
            .sort_values("sum_importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        grouped = grouped.copy()
        if "feature_group" not in grouped.columns and grouped.index.name == "feature_group":
            grouped = grouped.reset_index()
        if "sum" in grouped.columns:
            grouped = grouped.rename(columns={"sum": "sum_importance", "mean": "mean_importance"})
    summary = {
        "feature_count": len(fi),
        "nonzero_count": int((fi["importance"] > 0).sum()),
        "zero_count": int((fi["importance"] <= 0).sum()),
        "top_feature": fi.iloc[0]["feature"] if len(fi) else None,
        "top_feature_importance": fi.iloc[0]["importance"] if len(fi) else None,
    }
    return summary, fi, grouped


def summarize_pause_episodes(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out = {"episodes": len(df)}
    for col in ["duration_hours", "signals_blocked", "sl_blocked", "tp_blocked", "blocked_pnl_sum", "trigger_p_bad"]:
        if col in df.columns:
            out[f"{col}_mean"] = pd.to_numeric(df[col], errors="coerce").mean()
            out[f"{col}_median"] = pd.to_numeric(df[col], errors="coerce").median()
            out[f"{col}_max"] = pd.to_numeric(df[col], errors="coerce").max()
    return out


def summarize_monthly(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    pnl_col = "pnl_improvement" if "pnl_improvement" in df.columns else None
    out = {"months": len(df)}
    if pnl_col:
        vals = pd.to_numeric(df[pnl_col], errors="coerce")
        out["positive_months"] = int((vals > 0).sum())
        out["negative_months"] = int((vals < 0).sum())
        out["positive_months_ratio"] = float((vals > 0).mean())
        out["total_pnl_improvement"] = float(vals.sum())
    return out


def summarize_oos(raw_df: pd.DataFrame, scored_df: pd.DataFrame, accepted_df: pd.DataFrame, blocked_df: pd.DataFrame,
                  final_csv_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not raw_df.empty:
        out["raw_signals"] = len(raw_df)
        if "symbol" in raw_df.columns:
            out["raw_symbols"] = int(raw_df["symbol"].nunique())
        if "open_time" in raw_df.columns:
            out["raw_date_from"] = pd.to_datetime(raw_df["open_time"]).min()
            out["raw_date_to"] = pd.to_datetime(raw_df["open_time"]).max()
    if not scored_df.empty:
        out["scored_signals"] = len(scored_df)
        if "p_bad" in scored_df.columns:
            q = scored_df["p_bad"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
            out["p_bad_p50"] = q.get(0.5)
            out["p_bad_p90"] = q.get(0.9)
            out["p_bad_p95"] = q.get(0.95)
            out["p_bad_p99"] = q.get(0.99)
    if not accepted_df.empty:
        out["accepted_signals"] = len(accepted_df)
    if not blocked_df.empty:
        out["blocked_signals"] = len(blocked_df)
    if "accepted_signals" in out or "blocked_signals" in out:
        accepted = int(out.get("accepted_signals", 0))
        blocked = int(out.get("blocked_signals", 0))
        total = accepted + blocked
        out["blocked_share"] = blocked / total if total else np.nan
    if not final_csv_df.empty:
        out["final_csv_rows"] = len(final_csv_df)
    return out


def auto_findings(cv_summary: dict[str, Any], feature_summary: dict[str, Any], test_metrics: dict[str, Any],
                  oos_summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if cv_summary:
        vt = cv_summary.get("folds_valid", 0)
        no_op = cv_summary.get("folds_no_op", 0)
        if vt and no_op / vt >= 0.25:
            findings.append(f"Много no-op fold-ов: {no_op}/{vt}.")
        mean_pnl = cv_summary.get("mean_pnl_improvement")
        if mean_pnl is not None and not pd.isna(mean_pnl):
            if mean_pnl > 0:
                findings.append(f"Средний CV pnl_improvement положительный: {fmt(mean_pnl)}.")
            else:
                findings.append(f"Средний CV pnl_improvement отрицательный: {fmt(mean_pnl)}.")
        pos = cv_summary.get("folds_positive_pnl", 0)
        neg = cv_summary.get("folds_negative_pnl", 0)
        if vt:
            findings.append(f"Fold-ы с улучшением PnL: {pos}/{vt}; с ухудшением: {neg}/{vt}.")
        std = cv_summary.get("std_score")
        mean_score = cv_summary.get("mean_score")
        if std is not None and mean_score is not None and not pd.isna(std) and not pd.isna(mean_score):
            if abs(mean_score) > 0 and std > abs(mean_score) * 2:
                findings.append(f"CV нестабилен: std_score {fmt(std)} заметно больше mean_score {fmt(mean_score)}.")
    if feature_summary:
        zero_count = feature_summary.get("zero_count", 0)
        feature_count = feature_summary.get("feature_count", 0)
        if feature_count:
            findings.append(f"Нулевых feature importance: {zero_count}/{feature_count}.")
    if test_metrics:
        blocked_share = test_metrics.get("blocked_share")
        pnl_improvement = test_metrics.get("pnl_improvement")
        if blocked_share is not None and pnl_improvement is not None and not pd.isna(blocked_share) and not pd.isna(
                pnl_improvement):
            findings.append(
                f"Test/holdout: blocked_share={fmt_pct(blocked_share)}, pnl_improvement={fmt(pnl_improvement)}.")
    elif oos_summary:
        blocked_share = oos_summary.get("blocked_share")
        if blocked_share is not None and not pd.isna(blocked_share):
            findings.append(
                f"OOS export: blocked_share={fmt_pct(blocked_share)} ({fmt(oos_summary.get('blocked_signals', 0))}/{fmt((oos_summary.get('accepted_signals', 0) or 0) + (oos_summary.get('blocked_signals', 0) or 0))}).")
    return findings


def build_report(run_dir: Path) -> str:
    run_config = load_json(first_existing(run_dir, ["run_config.json"])) or {}
    dataset_manifest = load_json(first_existing(run_dir, ["dataset_manifest.json"])) or {}
    regime_builder_config = load_json(first_existing(run_dir, ["regime_builder_config.json"])) or {}
    liquid_universe = load_json(first_existing(run_dir, ["liquid_universe.json"])) or []
    best_model_params = load_json(first_existing(run_dir, ["best_model_params.json"])) or {}
    best_policy_params = load_json(first_existing(run_dir, ["best_policy_params.json"])) or {}
    cv_report = load_json(first_existing(run_dir, ["cv_report.json"])) or {}
    test_scorecard = load_json(first_existing(run_dir, ["test_scorecard.json"])) or {}
    test_metrics = load_json(first_existing(run_dir, ["test_metrics_full.json", "test_metrics.json"])) or {}
    if not test_metrics:
        test_metrics = test_scorecard.copy()
    elif test_scorecard:
        merged = test_metrics.copy()
        merged.update({k: v for k, v in test_scorecard.items() if k not in merged})
        test_metrics = merged

    fold_metrics = load_table(first_existing(run_dir, ["fold_metrics.csv"]))
    feature_importance = load_table(first_existing(run_dir, ["feature_importance.csv"]))
    feature_grouped = load_table(first_existing(run_dir, ["feature_importance_grouped.csv"]))
    monthly_backtest = load_table(first_existing(run_dir, ["monthly_backtest.csv"]))
    pause_episodes = load_table(first_existing(run_dir, ["pause_episodes.csv"]))
    p_bad_deciles = load_table(first_existing(run_dir, ["p_bad_deciles.csv"]))
    raw_oos = load_table(first_existing(run_dir, ["raw_detector_signals_oos.parquet"]))
    scored_oos = load_table(first_existing(run_dir, ["guard_scored_oos.parquet", "guard_scored_signals.parquet"]))
    accepted_oos = load_table(first_existing(run_dir, ["accepted_oos.parquet", "accepted_signals.parquet"]))
    blocked_oos = load_table(first_existing(run_dir, ["blocked_oos.parquet", "blocked_signals.parquet"]))
    final_oos_csv = load_table(first_existing(run_dir, ["final_guarded_signals_oos.csv", "final_signals.csv"]))
    regime_dataset = load_table(first_existing(run_dir, ["regime_dataset.parquet"]))

    target_col = run_config.get("target_col") or dataset_manifest.get("target_col")
    dataset_summary = summarize_dataset(regime_dataset, target_col)
    cv_summary, fold_metrics_valid = summarize_cv(fold_metrics, cv_report)
    feature_summary, feature_importance, feature_grouped = summarize_importance(feature_importance, feature_grouped)
    pause_summary = summarize_pause_episodes(pause_episodes)
    monthly_summary = summarize_monthly(monthly_backtest)
    oos_summary = summarize_oos(raw_oos, scored_oos, accepted_oos, blocked_oos, final_oos_csv)
    findings = auto_findings(cv_summary, feature_summary, test_metrics, oos_summary)

    lines: list[str] = []
    lines.append(f"# Regime Run Report: `{run_dir.name}`")
    lines.append("")
    lines.append(f"- Path: `{run_dir}`")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    if findings:
        lines.append("## Краткий вывод")
        lines.append("")
        for item in findings:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Конфиг")
    lines.append("")
    cfg_rows = [
        ["target_col", target_col or "—"],
        ["feature_profile", run_config.get("feature_profile") or dataset_manifest.get("feature_profile") or "—"],
        ["target_profile", run_config.get("target_profile") or dataset_manifest.get("target_profile") or "—"],
        ["trade_replay_source",
         dataset_manifest.get("trade_replay_source") or run_config.get("trade_replay_source") or "—"],
        ["train_end", run_config.get("train_end") or "—"],
        ["time_budget_min", run_config.get("time_budget_min") or "—"],
        ["fold_days", run_config.get("fold_days") or "—"],
        ["min_train_days", run_config.get("min_train_days") or "—"],
        ["max_blocked_share",
         fmt_pct(run_config.get("max_blocked_share")) if run_config.get("max_blocked_share") is not None else "—"],
        ["min_signal_keep_rate", fmt_pct(run_config.get("min_signal_keep_rate")) if run_config.get(
            "min_signal_keep_rate") is not None else "—"],
        ["policy_grid", run_config.get("policy_grid") or "—"],
        ["iterations", run_config.get("iterations") or "—"],
        ["n_features", run_config.get("n_features") or dataset_manifest.get("n_features") or "—"],
        ["n_samples", run_config.get("n_samples") or dataset_manifest.get("n_samples") or "—"],
    ]
    lines.append(md_table(["Параметр", "Значение"], cfg_rows))
    lines.append("")

    lines.append("## Датасет")
    lines.append("")
    ds_rows = [
        ["rows", fmt(dataset_summary.get("rows"))],
        ["date_from", fmt(dataset_summary.get("date_from"))],
        ["date_to", fmt(dataset_summary.get("date_to"))],
        ["symbols", fmt(dataset_summary.get("symbols"))],
        ["target_valid_rows", fmt(dataset_summary.get("target_valid_rows"))],
        ["target_rate", fmt_pct(dataset_summary.get("target_rate"))],
        ["top_n_universe", fmt(regime_builder_config.get("top_n_universe") or dataset_manifest.get("top_n_universe"))],
        ["high_lookback_bars",
         fmt(regime_builder_config.get("HIGH_8W_BARS") or dataset_manifest.get("high_lookback_bars"))],
        ["liquid_universe_size", fmt(len(liquid_universe)) if isinstance(liquid_universe, list) else "—"],
    ]
    lines.append(md_table(["Метрика", "Значение"], ds_rows))
    lines.append("")
    if isinstance(liquid_universe, list) and liquid_universe:
        if len(liquid_universe) <= 80:
            lines.append("**Liquid/fixed universe**")
            lines.append("")
            lines.append(", ".join(map(str, liquid_universe)))
            lines.append("")

    lines.append("## CV summary")
    lines.append("")
    if cv_summary:
        cv_rows = [
            ["folds_total", fmt(cv_summary.get("folds_total"))],
            ["folds_valid", fmt(cv_summary.get("folds_valid"))],
            ["folds_no_op", fmt(cv_summary.get("folds_no_op"))],
            ["folds_positive_pnl", fmt(cv_summary.get("folds_positive_pnl"))],
            ["folds_negative_pnl", fmt(cv_summary.get("folds_negative_pnl"))],
            ["mean_score", fmt(cv_summary.get("mean_score"))],
            ["std_score", fmt(cv_summary.get("std_score"))],
            ["median_score", fmt(cv_summary.get("median_score"))],
            ["mean_pnl_improvement", fmt(cv_summary.get("mean_pnl_improvement"))],
            ["median_pnl_improvement", fmt(cv_summary.get("median_pnl_improvement"))],
            ["mean_blocked_share", fmt_pct(cv_summary.get("mean_blocked_share"))],
            ["mean_signal_keep_rate", fmt_pct(cv_summary.get("mean_signal_keep_rate"))],
            ["mean_blocked_bad_precision", fmt(cv_summary.get("mean_blocked_bad_precision"))],
            ["mean_sl_capture", fmt(cv_summary.get("mean_sl_capture"))],
            ["mean_tp_tax", fmt(cv_summary.get("mean_tp_tax"))],
            ["mean_episode_recall", fmt(cv_summary.get("mean_episode_recall"))],
            ["mean_worst_window_improvement_12h", fmt(cv_summary.get("mean_worst_window_improvement_12h"))],
            ["mean_brier_score", fmt(cv_summary.get("mean_brier_score"))],
            ["mean_p_bad_p90", fmt(cv_summary.get("mean_p_bad_p90"))],
            ["mean_p_bad_p95", fmt(cv_summary.get("mean_p_bad_p95"))],
            ["mean_p_bad_p99", fmt(cv_summary.get("mean_p_bad_p99"))],
        ]
        lines.append(md_table(["Метрика", "Значение"], cv_rows))
        lines.append("")
        if not fold_metrics_valid.empty:
            fm = fold_metrics_valid.copy().reset_index(drop=True)
            fm["fold"] = np.arange(1, len(fm) + 1)
            best_idx = fm["score"].astype(float).idxmax()
            worst_idx = fm["score"].astype(float).idxmin()
            best_row = fm.loc[best_idx]
            worst_row = fm.loc[worst_idx]
            lines.append("### Лучший и худший fold")
            lines.append("")
            bw_rows = [
                [
                    "best",
                    fmt(best_row.get("fold")),
                    fmt(best_row.get("score")),
                    fmt(best_row.get("pnl_improvement")),
                    fmt_pct(best_row.get("blocked_share")),
                    fmt(best_row.get("blocked_bad_precision")),
                    fmt(best_row.get("sl_capture")),
                    fmt(best_row.get("tp_tax")),
                    fmt(best_row.get("p_bad_p99")),
                ],
                [
                    "worst",
                    fmt(worst_row.get("fold")),
                    fmt(worst_row.get("score")),
                    fmt(worst_row.get("pnl_improvement")),
                    fmt_pct(worst_row.get("blocked_share")),
                    fmt(worst_row.get("blocked_bad_precision")),
                    fmt(worst_row.get("sl_capture")),
                    fmt(worst_row.get("tp_tax")),
                    fmt(worst_row.get("p_bad_p99")),
                ],
            ]
            lines.append(md_table(
                ["Тип", "Fold", "Score", "PnL imp", "Blocked share", "Bad precision", "SL capture", "TP tax",
                 "p_bad_p99"], bw_rows))
            lines.append("")
            preview_cols = [c for c in ["fold", "score", "pnl_improvement", "blocked_share", "signal_keep_rate",
                                        "blocked_bad_precision", "sl_capture", "tp_tax", "worst_window_improvement_12h",
                                        "episode_recall", "p_bad_p95", "p_bad_p99"] if c in fm.columns]
            if preview_cols:
                preview = fm[preview_cols].copy()
                preview["fold"] = preview["fold"].astype(int)
                rows = []
                for _, row in preview.iterrows():
                    rows.append(
                        [fmt(row[c]) if c != "blocked_share" and c != "signal_keep_rate" else fmt_pct(row[c]) for c in
                         preview_cols])
                lines.append("### Fold table")
                lines.append("")
                lines.append(md_table(preview_cols, rows))
                lines.append("")
    else:
        lines.append("CV artifacts not found.")
        lines.append("")

    lines.append("## Best params")
    lines.append("")
    if best_model_params:
        model_rows = [[k, fmt(v)] for k, v in best_model_params.items()]
        lines.append("### Model")
        lines.append("")
        lines.append(md_table(["Параметр", "Значение"], model_rows))
        lines.append("")
    if best_policy_params:
        policy_rows = [[k, fmt(v)] for k, v in best_policy_params.items()]
        lines.append("### Policy")
        lines.append("")
        lines.append(md_table(["Параметр", "Значение"], policy_rows))
        lines.append("")

    lines.append("## Holdout / test")
    lines.append("")
    if test_metrics:
        ordered = [
            "signals_before", "signals_after", "pnl_before", "pnl_after", "pnl_improvement", "blocked_share",
            "signal_keep_rate",
            "blocked_bad_precision", "sl_capture", "tp_tax", "worst_window_improvement_12h", "episode_recall",
            "brier_score",
            "tp_blocked", "sl_blocked", "tp_kept", "sl_kept", "max_losing_streak_before", "max_losing_streak_after",
            "pause_episodes_count", "avg_pause_duration_hours", "p_bad_p50", "p_bad_p90", "p_bad_p95", "p_bad_p99",
        ]
        tm_rows = []
        for key in ordered:
            if key in test_metrics:
                value = test_metrics[key]
                if key in {"blocked_share", "signal_keep_rate"}:
                    value = fmt_pct(value)
                else:
                    value = fmt(value)
                tm_rows.append([key, value])
        if tm_rows:
            lines.append(md_table(["Метрика", "Значение"], tm_rows))
            lines.append("")
    else:
        lines.append("Holdout/test artifacts not found.")
        lines.append("")

    lines.append("## OOS export")
    lines.append("")
    if oos_summary:
        oos_rows = []
        for key in [
            "raw_signals", "raw_symbols", "raw_date_from", "raw_date_to", "scored_signals", "accepted_signals",
            "blocked_signals", "blocked_share",
            "final_csv_rows", "p_bad_p50", "p_bad_p90", "p_bad_p95", "p_bad_p99",
        ]:
            if key in oos_summary:
                val = fmt_pct(oos_summary[key]) if key == "blocked_share" else fmt(oos_summary[key])
                oos_rows.append([key, val])
        lines.append(md_table(["Метрика", "Значение"], oos_rows))
        lines.append("")
    else:
        lines.append("OOS export artifacts not found.")
        lines.append("")

    lines.append("## Feature importance")
    lines.append("")
    if feature_summary:
        fi_rows = [
            ["feature_count", fmt(feature_summary.get("feature_count"))],
            ["nonzero_count", fmt(feature_summary.get("nonzero_count"))],
            ["zero_count", fmt(feature_summary.get("zero_count"))],
            ["top_feature", fmt(feature_summary.get("top_feature"))],
            ["top_feature_importance", fmt(feature_summary.get("top_feature_importance"))],
        ]
        lines.append(md_table(["Метрика", "Значение"], fi_rows))
        lines.append("")
        if not feature_grouped.empty:
            fg = feature_grouped.copy()
            if "sum_importance" not in fg.columns and "sum" in fg.columns:
                fg = fg.rename(columns={"sum": "sum_importance", "mean": "mean_importance"})
            cols = [c for c in ["feature_group", "sum_importance", "mean_importance", "count"] if c in fg.columns]
            rows = []
            for _, row in fg[cols].head(12).iterrows():
                rows.append([fmt(row[c]) if c not in {"feature_group"} else row[c] for c in cols])
            lines.append("### Feature groups")
            lines.append("")
            lines.append(md_table(cols, rows))
            lines.append("")
        if not feature_importance.empty:
            top = feature_importance.head(20)
            rows = [[row["feature"], fmt(row["importance"]), feature_group(str(row["feature"]))] for _, row in
                    top.iterrows()]
            lines.append("### Top features")
            lines.append("")
            lines.append(md_table(["Feature", "Importance", "Group"], rows))
            lines.append("")
            dead = feature_importance[feature_importance["importance"] <= 0].head(25)
            if not dead.empty:
                rows = [[row["feature"], fmt(row["importance"]), feature_group(str(row["feature"]))] for _, row in
                        dead.iterrows()]
                lines.append("### Zero-importance features")
                lines.append("")
                lines.append(md_table(["Feature", "Importance", "Group"], rows))
                lines.append("")
    else:
        lines.append("Feature importance artifacts not found.")
        lines.append("")

    lines.append("## Pause episodes")
    lines.append("")
    if pause_summary:
        pe_rows = [[k, fmt(v)] for k, v in pause_summary.items()]
        lines.append(md_table(["Метрика", "Значение"], pe_rows))
        lines.append("")
        if not pause_episodes.empty:
            cols = [c for c in
                    ["episode_id", "start_time", "end_time", "duration_hours", "signals_blocked", "sl_blocked",
                     "tp_blocked", "blocked_pnl_sum", "trigger_p_bad", "resume_reason"] if c in pause_episodes.columns]
            rows = []
            for _, row in pause_episodes[cols].head(15).iterrows():
                rows.append([fmt(row[c]) for c in cols])
            lines.append("### First pause episodes")
            lines.append("")
            lines.append(md_table(cols, rows))
            lines.append("")
    else:
        lines.append("Pause episode artifacts not found.")
        lines.append("")

    lines.append("## Monthly backtest")
    lines.append("")
    if monthly_summary:
        mb_rows = [[k, fmt_pct(v) if k == "positive_months_ratio" else fmt(v)] for k, v in monthly_summary.items()]
        lines.append(md_table(["Метрика", "Значение"], mb_rows))
        lines.append("")
        if not monthly_backtest.empty:
            sort_col = "pnl_improvement" if "pnl_improvement" in monthly_backtest.columns else monthly_backtest.columns[
                -1]
            cols = [c for c in
                    ["month", "signals", "signals_after", "pnl_before", "pnl_after", "pnl_improvement", "blocked_share",
                     "sl_blocked", "tp_blocked"] if c in monthly_backtest.columns]
            best = monthly_backtest.sort_values(sort_col, ascending=False).head(3)
            worst = monthly_backtest.sort_values(sort_col, ascending=True).head(3)
            rows = []
            for tag, df in [("best", best), ("worst", worst)]:
                for _, row in df[cols].iterrows():
                    vals = [fmt_pct(row[c]) if c == "blocked_share" else fmt(row[c]) for c in cols]
                    rows.append([tag] + vals)
            lines.append(md_table(["Type"] + cols, rows))
            lines.append("")
    else:
        lines.append("Monthly backtest artifacts not found.")
        lines.append("")

    lines.append("## p_bad deciles")
    lines.append("")
    if not p_bad_deciles.empty:
        dec = p_bad_deciles.copy()
        sort_col = "mean_p_bad" if "mean_p_bad" in dec.columns else dec.columns[0]
        dec = dec.sort_values(sort_col, ascending=False)
        cols = [c for c in
                ["decile", "count", "mean_p_bad", "bad_rate", "future_block_value_12h_mean", "future_pnl_sum_12h_mean",
                 "sl_rate_next_5_mean"] if c in dec.columns]
        rows = []
        for _, row in dec.head(3).iterrows():
            rows.append([fmt(row[c]) for c in cols])
        lines.append(md_table(cols, rows))
        lines.append("")
    else:
        lines.append("p_bad deciles artifacts not found.")
        lines.append("")

    lines.append("## Files found")
    lines.append("")
    file_rows = []
    for name in KEY_FILES:
        path = run_dir / name
        if path.exists():
            size = path.stat().st_size
            file_rows.append([name, "yes", fmt(size)])
    if file_rows:
        lines.append(md_table(["File", "Exists", "Bytes"], file_rows))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    report = build_report(run_dir)
    out_path = Path(args.output) if args.output else run_dir / "run_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
