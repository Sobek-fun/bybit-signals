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
    "splits.json",
    "feature_columns.json",
    "best_params.json",
    "best_threshold.json",
    "cv_report.json",
    "folds.json",
    "fold_metrics.csv",
    "leaderboard.csv",
    "feature_importance.csv",
    "feature_importance_grouped.csv",
    "threshold_sweep.csv",
    "calibration_sweep_val_live_shadow.csv",
    "calibration_sweep_val_eventcentric.csv",
    "calibration_sweep_val.csv",
    "metrics_val.json",
    "metrics_holdout_live_shadow.json",
    "metrics_holdout_live_shadow_val.json",
    "metrics_eventcentric_test.json",
    "metrics_test.json",
    "predictions_val.parquet",
    "predictions_test.parquet",
    "predicted_signals_val.csv",
    "predicted_signals_holdout.csv",
    "predicted_signals_eventcentric_test.csv",
    "cv_oof_signals_verbose.parquet",
    "cv_oos_signals_verbose.parquet",
    "holdout_window_summary_6h.csv",
    "holdout_symbol_summary.csv",
    "feature_na_report.csv",
    "pump_labels_filtered.csv",
    "training_points.parquet",
    "features.parquet",
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


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def normalize_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def flatten_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}

    event_level = metrics.get("event_level", {}) if isinstance(metrics.get("event_level"), dict) else {}
    point_level = metrics.get("point_level", {}) if isinstance(metrics.get("point_level"), dict) else {}
    trade_quality = metrics.get("trade_quality", {}) if isinstance(metrics.get("trade_quality"), dict) else {}
    signal_quality = metrics.get("signal_quality", {}) if isinstance(metrics.get("signal_quality"), dict) else {}

    flat = {}
    for src in (metrics, event_level, point_level):
        for k, v in src.items():
            if isinstance(v, dict):
                continue
            flat[k] = v

    flat["trade_quality_score"] = metrics.get("trade_quality_score")

    for horizon in (16, 32, 48):
        mfe_key = f"mfe_short_{horizon}"
        mae_key = f"mae_short_{horizon}"
        mfe = trade_quality.get(mfe_key, {}) if isinstance(trade_quality.get(mfe_key), dict) else {}
        mae = trade_quality.get(mae_key, {}) if isinstance(trade_quality.get(mae_key), dict) else {}
        if mfe:
            flat[f"{mfe_key}_median"] = mfe.get("median")
            flat[f"{mfe_key}_pct_above_2pct"] = mfe.get("pct_above_2pct")
            flat[f"{mfe_key}_count"] = mfe.get("count")
        if mae:
            flat[f"{mae_key}_median"] = mae.get("median")
            flat[f"{mae_key}_count"] = mae.get("count")

    for k, v in signal_quality.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            flat[k] = v

    return flat


def feature_group(name: str) -> str:
    if name.startswith(("ret_", "cum_ret", "drawdown")):
        return "returns"
    if name.startswith(("rsi", "mfi", "macd", "atr", "bb_", "vwap", "obv")):
        return "indicators"
    if name.startswith(("runup", "pump_", "predump", "strong_cond", "vol_fade", "rsi_fade", "macd_fade")):
        return "pump_detector"
    if name.startswith(("dist_to_", "liq_", "pdh", "pwh", "eqh", "sweep_", "overshoot_", "touched_")):
        return "liquidity"
    if name.startswith(("btc_", "rel_")):
        return "market"
    if name.startswith(("upper_wick", "lower_wick", "wick_", "body_ratio", "close_pos", "range", "count_red")):
        return "candle"
    if name.startswith(("vol_", "volume", "log_volume")):
        return "volume"
    return "other"


# --------------------------------------------------------------------------------------
# Dataset and split summary
# --------------------------------------------------------------------------------------
def summarize_dataset(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(first_existing(run_dir, ["dataset_manifest.json"])) or {}
    summary: dict[str, Any] = {}

    if manifest:
        summary.update({
            "dataset_parquet": manifest.get("dataset_parquet") or manifest.get("path"),
            "labels_path": manifest.get("labels_path") or manifest.get("labels"),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "neg_before": manifest.get("neg_before"),
            "neg_after": manifest.get("neg_after"),
            "pos_offsets": manifest.get("pos_offsets"),
            "include_b": manifest.get("include_b"),
            "window_bars": manifest.get("window_bars"),
            "warmup_bars": manifest.get("warmup_bars"),
            "feature_set": manifest.get("feature_set"),
            "n_rows": manifest.get("n_rows") or manifest.get("rows"),
            "n_events": manifest.get("n_events"),
            "n_a_events": manifest.get("n_a_events"),
            "n_b_events": manifest.get("n_b_events"),
            "feature_count": manifest.get("feature_count") or manifest.get("n_features"),
        })

    training_points = load_table(first_existing(run_dir, ["training_points.parquet"]))
    if not training_points.empty:
        tp = training_points.copy()
        if "open_time" in tp.columns:
            tp["open_time"] = normalize_dt(tp["open_time"])
        summary.setdefault("n_rows", len(tp))
        summary.setdefault("n_events", int(tp["event_id"].nunique()) if "event_id" in tp.columns else None)
        if "pump_la_type" in tp.columns:
            summary.setdefault("n_a_events", int(tp.loc[tp["pump_la_type"] == "A", "event_id"].nunique()))
            summary.setdefault("n_b_events", int(tp.loc[tp["pump_la_type"] == "B", "event_id"].nunique()))
        if "open_time" in tp.columns:
            summary.setdefault("date_from", tp["open_time"].min())
            summary.setdefault("date_to", tp["open_time"].max())

    features = load_table(first_existing(run_dir, ["features.parquet"]))
    if not features.empty:
        fx = features.copy()
        if "open_time" in fx.columns:
            fx["open_time"] = normalize_dt(fx["open_time"])
            summary.setdefault("date_from", fx["open_time"].min())
            summary.setdefault("date_to", fx["open_time"].max())
        if "event_id" in fx.columns:
            summary.setdefault("n_events", int(fx["event_id"].nunique()))
        excluded = {
            "event_id", "symbol", "open_time", "offset", "y", "pump_la_type", "runup_pct", "split", "target",
            "timeframe", "window_bars", "warmup_bars", "sample_weight",
        }
        feature_cols = [c for c in fx.columns if c not in excluded]
        summary.setdefault("feature_count", len(feature_cols))
        if "y" in fx.columns:
            summary.setdefault("positive_rate", float(pd.to_numeric(fx["y"], errors="coerce").mean()))
        if "sample_weight" in fx.columns:
            w = pd.to_numeric(fx["sample_weight"], errors="coerce")
            summary.setdefault("sample_weight_mean", float(w.mean()))
            summary.setdefault("sample_weight_p95", float(w.quantile(0.95)))

    return summary


def summarize_splits(run_dir: Path) -> dict[str, Any]:
    splits = load_json(first_existing(run_dir, ["splits.json"])) or {}
    out: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split = splits.get(split_name, {}) if isinstance(splits.get(split_name), dict) else {}
        if split:
            out[f"{split_name}_events"] = split.get("n_events")
            out[f"{split_name}_points"] = split.get("n_points")
            out[f"{split_name}_positive"] = split.get("n_positive")
            out[f"{split_name}_negative"] = split.get("n_negative")
    return out


# --------------------------------------------------------------------------------------
# CV / calibration summary
# --------------------------------------------------------------------------------------
def summarize_cv(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    cv_report = load_json(first_existing(run_dir, ["cv_report.json"])) or {}
    fold_metrics = load_table(first_existing(run_dir, ["fold_metrics.csv"]))
    if fold_metrics.empty and isinstance(cv_report.get("fold_results"), list):
        fold_metrics = pd.DataFrame(cv_report["fold_results"])

    if fold_metrics.empty:
        return {}, fold_metrics

    fm = fold_metrics.copy()
    valid_mask = fm.get("valid", pd.Series([True] * len(fm), index=fm.index)).fillna(True).astype(bool)
    valid = fm[valid_mask].copy()
    if valid.empty:
        valid = fm.copy()

    out = {
        "folds_total": int(len(fm)),
        "folds_valid": int(len(valid)),
    }

    for col in ("score", "threshold", "min_pending_bars", "drop_delta", "hit0_rate", "hit1_rate", "early_rate", "late_rate", "miss_rate", "fp_b_rate", "signal_count", "signals_total", "median_pred_offset"):
        if col in valid.columns:
            vals = pd.to_numeric(valid[col], errors="coerce")
            out[f"mean_{col}"] = float(vals.mean())
            out[f"std_{col}"] = float(vals.std(ddof=0))

    if "score" in valid.columns:
        vals = pd.to_numeric(valid["score"], errors="coerce")
        out["positive_score_folds"] = int((vals > 0).sum())
        out["negative_score_folds"] = int((vals < 0).sum())

    return out, valid.reset_index(drop=True)


def summarize_calibration(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    best_threshold = load_json(first_existing(run_dir, ["best_threshold.json"])) or {}
    sweep = load_table(first_existing(run_dir, ["calibration_sweep_val_live_shadow.csv", "calibration_sweep_val.csv", "threshold_sweep.csv"]))

    out = {
        "threshold": best_threshold.get("threshold"),
        "signal_rule": best_threshold.get("signal_rule"),
        "min_pending_bars": best_threshold.get("min_pending_bars"),
        "drop_delta": best_threshold.get("drop_delta"),
        "abstain_margin": best_threshold.get("abstain_margin"),
    }

    if not sweep.empty:
        sw = sweep.copy()
        if "score" in sw.columns:
            sw["score"] = pd.to_numeric(sw["score"], errors="coerce")
            best_idx = sw["score"].idxmax()
            if pd.notna(best_idx):
                best_row = sw.loc[best_idx]
                for col in ("score", "hit0_rate", "hit1_rate", "early_rate", "late_rate", "miss_rate", "fp_b_rate", "avg_offset", "median_offset", "signal_count", "signals_total"):
                    if col in best_row:
                        out[f"best_{col}"] = best_row[col]

    return out, sweep


# --------------------------------------------------------------------------------------
# Holdout / signal summaries
# --------------------------------------------------------------------------------------
def _max_losing_streak(outcomes: list[str]) -> int:
    streak = 0
    max_streak = 0
    for outcome in outcomes:
        if str(outcome).upper() == "SL":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _worst_window(df: pd.DataFrame, hours: int, pnl_col: str) -> float | None:
    if df.empty or "open_time" not in df.columns or pnl_col not in df.columns:
        return None
    work = df[["open_time", pnl_col]].copy()
    work = work.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    if work.empty:
        return None
    work[pnl_col] = pd.to_numeric(work[pnl_col], errors="coerce").fillna(0.0)
    worst = 0.0
    window = pd.Timedelta(hours=hours)
    for i in range(len(work)):
        start = work.iloc[i]["open_time"]
        end = start + window
        mask = (work["open_time"] >= start) & (work["open_time"] < end)
        total = float(work.loc[mask, pnl_col].sum())
        worst = min(worst, total)
    return worst


def summarize_metrics_json(run_dir: Path, split_name: str) -> dict[str, Any]:
    metrics = load_json(first_existing(run_dir, [f"metrics_{split_name}.json", f"signal_metrics_{split_name}.json"])) or {}
    return flatten_metrics(metrics)


def summarize_signals_df(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    for col in ("open_time", "entry_time", "exit_time", "peak_open_time"):
        if col in work.columns:
            work[col] = normalize_dt(work[col])

    out: dict[str, Any] = {
        "signals": int(len(work)),
        "symbols": int(work["symbol"].nunique()) if "symbol" in work.columns else None,
    }
    if "open_time" in work.columns:
        out["date_from"] = work["open_time"].min()
        out["date_to"] = work["open_time"].max()
        span_days = (work["open_time"].max() - work["open_time"].min()).total_seconds() / 86400 if len(work) > 1 else np.nan
        if span_days and span_days > 0:
            out["signals_per_30d"] = float(len(work) / span_days * 30)

    if "signal_offset" in work.columns:
        offsets = pd.to_numeric(work["signal_offset"], errors="coerce")
        out["signal_offset_p50"] = float(offsets.median()) if offsets.notna().any() else None
        out["signal_offset_p90"] = float(offsets.quantile(0.90)) if offsets.notna().any() else None
        out["early_signals"] = int((offsets < 0).sum())
        out["late_signals"] = int((offsets > 1).sum())
        out["hit0_signals"] = int((offsets == 0).sum())
        out["hit01_signals"] = int(offsets.isin([0, 1]).sum())
        out["early_share"] = float((offsets < 0).mean()) if len(offsets) else None
        out["late_share"] = float((offsets > 1).mean()) if len(offsets) else None

    if "event_type" in work.columns:
        et = work["event_type"].astype(str)
        out["signals_on_A"] = int((et == "A").sum())
        out["signals_on_B"] = int((et == "B").sum())
        out["b_signal_share"] = float((et == "B").mean()) if len(et) else None

    pnl_col = None
    for candidate in ("trade_pnl_pct", "pnl_pct"):
        if candidate in work.columns:
            pnl_col = candidate
            work[candidate] = pd.to_numeric(work[candidate], errors="coerce")
            break

    monthly = pd.DataFrame()
    worst_symbols = pd.DataFrame()
    worst_6h = pd.DataFrame()

    if "trade_outcome" in work.columns:
        outcomes = work["trade_outcome"].astype(str).str.upper()
        out["tp"] = int((outcomes == "TP").sum())
        out["sl"] = int((outcomes == "SL").sum())
        out["timeout"] = int((outcomes == "TIMEOUT").sum())
        out["ambiguous"] = int((outcomes == "AMBIGUOUS").sum())
        out["unknown"] = int((outcomes == "UNKNOWN").sum())
        resolved_mask = outcomes.isin(["TP", "SL"])
        out["resolved"] = int(resolved_mask.sum())
        if resolved_mask.any():
            out["tp_rate_resolved"] = float((outcomes[resolved_mask] == "TP").mean())
            out["sl_rate_resolved"] = float((outcomes[resolved_mask] == "SL").mean())
        out["max_losing_streak"] = _max_losing_streak(outcomes.tolist())

    if pnl_col:
        pnl = work[pnl_col].fillna(0.0)
        out["pnl_sum"] = float(pnl.sum())
        out["expectancy_all"] = float(pnl.mean()) if len(pnl) else None
        if "trade_outcome" in work.columns:
            outcomes = work["trade_outcome"].astype(str).str.upper()
            resolved_mask = outcomes.isin(["TP", "SL"])
            if resolved_mask.any():
                out["expectancy_resolved"] = float(work.loc[resolved_mask, pnl_col].mean())
            gross_profit = float(work.loc[work[pnl_col] > 0, pnl_col].sum())
            gross_loss = float(-work.loc[work[pnl_col] < 0, pnl_col].sum())
            out["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else np.nan
        out["worst_6h_pnl"] = _worst_window(work, 6, pnl_col)
        out["worst_24h_pnl"] = _worst_window(work, 24, pnl_col)

    if pnl_col and "open_time" in work.columns:
        temp = work.copy()
        temp["month"] = temp["open_time"].dt.to_period("M").astype(str)
        if "trade_outcome" in temp.columns:
            grouped = temp.groupby("month", as_index=False).agg(
                signals=("symbol", "count"),
                pnl_sum=(pnl_col, "sum"),
                tp=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "TP").sum())),
                sl=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "SL").sum())),
                timeout=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "TIMEOUT").sum())),
            )
            grouped["expectancy"] = grouped["pnl_sum"] / grouped["signals"].replace(0, np.nan)
            monthly = grouped.sort_values("month").reset_index(drop=True)

    if pnl_col and "symbol" in work.columns:
        ws = work.groupby("symbol", as_index=False).agg(
            signals=("symbol", "count"),
            pnl_sum=(pnl_col, "sum"),
        )
        if "trade_outcome" in work.columns:
            per_symbol = work.groupby("symbol")["trade_outcome"].apply(lambda s: int((s.astype(str).str.upper() == "SL").sum()))
            ws["sl"] = ws["symbol"].map(per_symbol)
        ws["expectancy"] = ws["pnl_sum"] / ws["signals"].replace(0, np.nan)
        worst_symbols = ws.sort_values(["pnl_sum", "signals"], ascending=[True, False]).reset_index(drop=True)

    if pnl_col and "open_time" in work.columns:
        temp = work.copy()
        temp["bucket_6h"] = temp["open_time"].dt.floor("6h")
        agg = {"signals": ("symbol", "count"), "pnl_sum": (pnl_col, "sum")}
        if "trade_outcome" in temp.columns:
            grouped = temp.groupby("bucket_6h", as_index=False).agg(
                signals=("symbol", "count"),
                pnl_sum=(pnl_col, "sum"),
                tp=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "TP").sum())),
                sl=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "SL").sum())),
                timeout=("trade_outcome", lambda s: int((s.astype(str).str.upper() == "TIMEOUT").sum())),
            )
            grouped["expectancy"] = grouped["pnl_sum"] / grouped["signals"].replace(0, np.nan)
            worst_6h = grouped.sort_values("pnl_sum", ascending=True).reset_index(drop=True)

    return out, monthly, worst_symbols, worst_6h


def summarize_holdout(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_flat = summarize_metrics_json(run_dir, "holdout_live_shadow")
    if not metrics_flat:
        metrics_flat = summarize_metrics_json(run_dir, "test")
    signals = load_table(first_existing(run_dir, ["predicted_signals_holdout.csv", "predicted_signals_holdout.parquet"]))
    signal_summary, monthly, worst_symbols, worst_6h = summarize_signals_df(signals)

    merged = metrics_flat.copy()
    for k, v in signal_summary.items():
        merged.setdefault(k, v)

    return merged, signals, monthly, worst_symbols, worst_6h


def summarize_eventcentric(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics_flat = summarize_metrics_json(run_dir, "eventcentric_test")
    if not metrics_flat:
        metrics_flat = summarize_metrics_json(run_dir, "test")
    signals = load_table(first_existing(run_dir, ["predicted_signals_eventcentric_test.csv", "predicted_signals_holdout.csv"]))
    signal_summary, _monthly, _worst_symbols, _worst_6h = summarize_signals_df(signals)
    merged = metrics_flat.copy()
    for k, v in signal_summary.items():
        merged.setdefault(k, v)
    return merged, signals


def summarize_val(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics_flat = summarize_metrics_json(run_dir, "val")
    signals = load_table(first_existing(run_dir, ["predicted_signals_val.csv", "predicted_signals_val.parquet"]))
    signal_summary, _monthly, _worst_symbols, _worst_6h = summarize_signals_df(signals)
    merged = metrics_flat.copy()
    for k, v in signal_summary.items():
        merged.setdefault(k, v)
    return merged, signals


# --------------------------------------------------------------------------------------
# Feature importance summary
# --------------------------------------------------------------------------------------
def summarize_importance(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fi = load_table(first_existing(run_dir, ["feature_importance.csv"]))
    grouped = load_table(first_existing(run_dir, ["feature_importance_grouped.csv"]))
    if fi.empty:
        return {}, fi, grouped

    work = fi.copy()
    work["importance"] = pd.to_numeric(work["importance"], errors="coerce").fillna(0.0)
    work = work.sort_values("importance", ascending=False).reset_index(drop=True)

    if grouped.empty:
        tmp = work.copy()
        tmp["feature_group"] = tmp["feature"].astype(str).map(feature_group)
        grouped = (
            tmp.groupby("feature_group", as_index=False)
            .agg(sum_importance=("importance", "sum"), mean_importance=("importance", "mean"), count=("feature", "count"))
            .sort_values("sum_importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        grouped = grouped.copy()
        if "group" in grouped.columns and "feature_group" not in grouped.columns:
            grouped = grouped.rename(columns={"group": "feature_group"})
        if "total_importance" in grouped.columns and "sum_importance" not in grouped.columns:
            grouped = grouped.rename(columns={"total_importance": "sum_importance", "mean_importance": "mean_importance", "feature_count": "count"})
        if "sum" in grouped.columns and "sum_importance" not in grouped.columns:
            grouped = grouped.rename(columns={"sum": "sum_importance", "mean": "mean_importance"})

    summary = {
        "feature_count": int(len(work)),
        "nonzero_count": int((work["importance"] > 0).sum()),
        "zero_count": int((work["importance"] <= 0).sum()),
        "top_feature": work.iloc[0]["feature"] if len(work) else None,
        "top_feature_importance": work.iloc[0]["importance"] if len(work) else None,
    }

    return summary, work, grouped


# --------------------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------------------
def auto_findings(
    dataset_summary: dict[str, Any],
    cv_summary: dict[str, Any],
    calibration_summary: dict[str, Any],
    val_summary: dict[str, Any],
    holdout_summary: dict[str, Any],
    feature_summary: dict[str, Any],
) -> list[str]:
    findings: list[str] = []

    n_b_events = dataset_summary.get("n_b_events")
    if n_b_events is not None and not pd.isna(n_b_events) and int(n_b_events) == 0:
        findings.append("В датасете нет B-событий: метрика false positive по B невалидна или бесполезна.")

    cv_folds = cv_summary.get("folds_valid")
    if cv_folds:
        std_score = cv_summary.get("std_score") or cv_summary.get("std_mean_score")
        mean_score = cv_summary.get("mean_score") or cv_summary.get("mean_mean_score")
        if std_score is not None and mean_score is not None:
            try:
                if not pd.isna(std_score) and not pd.isna(mean_score) and abs(float(mean_score)) > 0 and float(std_score) > abs(float(mean_score)) * 1.5:
                    findings.append(f"CV нестабилен: std_score={fmt(std_score)} при mean_score={fmt(mean_score)}.")
            except Exception:
                pass

    early_rate = holdout_summary.get("early_rate")
    if early_rate is not None and not pd.isna(early_rate) and float(early_rate) >= 0.20:
        findings.append(f"На holdout слишком много ранних сигналов: early_rate={fmt_pct(early_rate)}.")

    miss_rate = holdout_summary.get("miss_rate")
    if miss_rate is not None and not pd.isna(miss_rate) and float(miss_rate) >= 0.50:
        findings.append(f"На holdout модель часто молчит: miss_rate={fmt_pct(miss_rate)}.")

    fp_b_rate = holdout_summary.get("fp_b_rate")
    if fp_b_rate is not None and not pd.isna(fp_b_rate) and float(fp_b_rate) >= 0.20:
        findings.append(f"На holdout высокий false positive по B: fp_b_rate={fmt_pct(fp_b_rate)}.")

    signals = holdout_summary.get("signals")
    if signals is not None and not pd.isna(signals) and int(signals) < 20:
        findings.append(f"На holdout мало сигналов ({fmt(signals)}), выводы по качеству будут шумными.")

    sl_rate_resolved = holdout_summary.get("sl_rate_resolved")
    if sl_rate_resolved is not None and not pd.isna(sl_rate_resolved) and float(sl_rate_resolved) >= 0.60:
        findings.append(f"На holdout SL-rate по resolved сигналам высокий: {fmt_pct(sl_rate_resolved)}.")

    worst_24h = holdout_summary.get("worst_24h_pnl")
    if worst_24h is not None and not pd.isna(worst_24h) and float(worst_24h) < 0:
        findings.append(f"Худшее 24h окно на holdout отрицательное: {fmt(worst_24h)}.")

    if feature_summary:
        zero_count = feature_summary.get("zero_count")
        feature_count = feature_summary.get("feature_count")
        if zero_count is not None and feature_count:
            try:
                if feature_count and zero_count / feature_count >= 0.35:
                    findings.append(f"Слишком много zero-importance фич: {zero_count}/{feature_count}.")
            except Exception:
                pass

    val_hit = val_summary.get("hit0_or_hit1_rate")
    test_hit = holdout_summary.get("hit0_or_hit1_rate")
    if val_hit is not None and test_hit is not None:
        try:
            if not pd.isna(val_hit) and not pd.isna(test_hit) and float(test_hit) + 0.10 < float(val_hit):
                findings.append(
                    f"Есть заметная просадка от val к holdout по hit0_or_hit1_rate: {fmt_pct(val_hit)} -> {fmt_pct(test_hit)}."
                )
        except Exception:
            pass

    threshold = calibration_summary.get("threshold")
    if threshold is not None and not pd.isna(threshold):
        findings.append(
            f"Финальный rule: threshold={fmt(threshold)}, min_pending_bars={fmt(calibration_summary.get('min_pending_bars'))}, drop_delta={fmt(calibration_summary.get('drop_delta'))}."
        )

    return findings


# --------------------------------------------------------------------------------------
# Report builder
# --------------------------------------------------------------------------------------
def build_report(run_dir: Path) -> str:
    run_config = load_json(first_existing(run_dir, ["run_config.json"])) or {}
    dataset_summary = summarize_dataset(run_dir)
    split_summary = summarize_splits(run_dir)
    cv_summary, fold_metrics = summarize_cv(run_dir)
    calibration_summary, sweep_df = summarize_calibration(run_dir)
    val_summary, val_signals = summarize_val(run_dir)
    holdout_summary, holdout_signals, monthly_df, worst_symbols_df, worst_6h_df = summarize_holdout(run_dir)
    eventcentric_summary, _eventcentric_signals = summarize_eventcentric(run_dir)
    feature_summary, feature_importance, feature_grouped = summarize_importance(run_dir)

    findings = auto_findings(
        dataset_summary=dataset_summary,
        cv_summary=cv_summary,
        calibration_summary=calibration_summary,
        val_summary=val_summary,
        holdout_summary=holdout_summary,
        feature_summary=feature_summary,
    )

    lines: list[str] = []
    lines.append("# Threshold run report")
    lines.append("")
    lines.append(f"- Path: `{run_dir}`")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    if findings:
        lines.append("## Key findings")
        lines.append("")
        for item in findings:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Run config")
    lines.append("")
    cfg_rows = [
        ["mode", run_config.get("mode") or "—"],
        ["dataset_parquet", run_config.get("dataset_parquet") or dataset_summary.get("dataset_parquet") or "—"],
        ["train_end", run_config.get("train_end") or "—"],
        ["val_end", run_config.get("val_end") or "—"],
        ["test_end", run_config.get("test_end") or "—"],
        ["signal_rule", calibration_summary.get("signal_rule") or run_config.get("signal_rule") or "—"],
        ["tune_strategy", run_config.get("tune_strategy") or "—"],
        ["time_budget_min", fmt(run_config.get("time_budget_min"))],
        ["fold_months", fmt(run_config.get("fold_months"))],
        ["min_train_months", fmt(run_config.get("min_train_months"))],
        ["embargo_bars", fmt(run_config.get("embargo_bars"))],
        ["iterations", fmt(run_config.get("iterations"))],
        ["prune_features", fmt(run_config.get("prune_features"))],
    ]
    lines.append(md_table(["Parameter", "Value"], cfg_rows))
    lines.append("")

    lines.append("## Dataset and splits")
    lines.append("")
    ds_rows = [
        ["date_from", fmt(dataset_summary.get("date_from"))],
        ["date_to", fmt(dataset_summary.get("date_to"))],
        ["n_rows", fmt(dataset_summary.get("n_rows"))],
        ["n_events", fmt(dataset_summary.get("n_events"))],
        ["n_a_events", fmt(dataset_summary.get("n_a_events"))],
        ["n_b_events", fmt(dataset_summary.get("n_b_events"))],
        ["positive_rate", fmt_pct(dataset_summary.get("positive_rate"))],
        ["feature_count", fmt(dataset_summary.get("feature_count"))],
        ["window_bars", fmt(dataset_summary.get("window_bars"))],
        ["warmup_bars", fmt(dataset_summary.get("warmup_bars"))],
        ["feature_set", fmt(dataset_summary.get("feature_set"))],
        ["neg_before", fmt(dataset_summary.get("neg_before"))],
        ["neg_after", fmt(dataset_summary.get("neg_after"))],
        ["pos_offsets", fmt(dataset_summary.get("pos_offsets"))],
        ["include_b", fmt(dataset_summary.get("include_b"))],
        ["sample_weight_mean", fmt(dataset_summary.get("sample_weight_mean"))],
        ["sample_weight_p95", fmt(dataset_summary.get("sample_weight_p95"))],
        ["train_events", fmt(split_summary.get("train_events"))],
        ["val_events", fmt(split_summary.get("val_events"))],
        ["test_events", fmt(split_summary.get("test_events"))],
    ]
    lines.append(md_table(["Metric", "Value"], ds_rows))
    lines.append("")

    lines.append("## CV robustness")
    lines.append("")
    if cv_summary:
        cv_rows = [
            ["folds_total", fmt(cv_summary.get("folds_total"))],
            ["folds_valid", fmt(cv_summary.get("folds_valid"))],
            ["positive_score_folds", fmt(cv_summary.get("positive_score_folds"))],
            ["negative_score_folds", fmt(cv_summary.get("negative_score_folds"))],
            ["mean_score", fmt(cv_summary.get("mean_score") or cv_summary.get("mean_mean_score"))],
            ["std_score", fmt(cv_summary.get("std_score") or cv_summary.get("std_mean_score"))],
            ["mean_threshold", fmt(cv_summary.get("mean_threshold"))],
            ["mean_min_pending_bars", fmt(cv_summary.get("mean_min_pending_bars"))],
            ["mean_drop_delta", fmt(cv_summary.get("mean_drop_delta"))],
            ["mean_hit0_rate", fmt_pct(cv_summary.get("mean_hit0_rate"))],
            ["mean_hit1_rate", fmt_pct(cv_summary.get("mean_hit1_rate"))],
            ["mean_early_rate", fmt_pct(cv_summary.get("mean_early_rate"))],
            ["mean_late_rate", fmt_pct(cv_summary.get("mean_late_rate"))],
            ["mean_miss_rate", fmt_pct(cv_summary.get("mean_miss_rate"))],
            ["mean_fp_b_rate", fmt_pct(cv_summary.get("mean_fp_b_rate"))],
            ["mean_signal_count", fmt(cv_summary.get("mean_signal_count") or cv_summary.get("mean_signals_total"))],
            ["mean_median_pred_offset", fmt(cv_summary.get("mean_median_pred_offset"))],
        ]
        lines.append(md_table(["Metric", "Value"], cv_rows))
        lines.append("")

        if not fold_metrics.empty:
            preview_cols = [
                c for c in [
                    "fold_idx", "score", "threshold", "min_pending_bars", "drop_delta",
                    "hit0_rate", "hit1_rate", "early_rate", "late_rate", "miss_rate", "fp_b_rate",
                    "signal_count", "signals_total", "median_pred_offset"
                ] if c in fold_metrics.columns
            ]
            if preview_cols:
                rows: list[list[Any]] = []
                for _, row in fold_metrics[preview_cols].iterrows():
                    display_row = []
                    for col in preview_cols:
                        value = row[col]
                        if col.endswith("_rate"):
                            display_row.append(fmt_pct(value))
                        else:
                            display_row.append(fmt(value))
                    rows.append(display_row)
                lines.append("### Fold table")
                lines.append("")
                lines.append(md_table(preview_cols, rows))
                lines.append("")
    else:
        lines.append("CV artifacts not found.")
        lines.append("")

    lines.append("## Val calibration")
    lines.append("")
    calib_rows = [
        ["threshold", fmt(calibration_summary.get("threshold"))],
        ["signal_rule", fmt(calibration_summary.get("signal_rule"))],
        ["min_pending_bars", fmt(calibration_summary.get("min_pending_bars"))],
        ["drop_delta", fmt(calibration_summary.get("drop_delta"))],
        ["abstain_margin", fmt(calibration_summary.get("abstain_margin"))],
        ["best_score", fmt(calibration_summary.get("best_score"))],
        ["best_hit0_rate", fmt_pct(calibration_summary.get("best_hit0_rate"))],
        ["best_hit1_rate", fmt_pct(calibration_summary.get("best_hit1_rate"))],
        ["best_early_rate", fmt_pct(calibration_summary.get("best_early_rate"))],
        ["best_late_rate", fmt_pct(calibration_summary.get("best_late_rate"))],
        ["best_miss_rate", fmt_pct(calibration_summary.get("best_miss_rate"))],
        ["best_fp_b_rate", fmt_pct(calibration_summary.get("best_fp_b_rate"))],
        ["best_median_offset", fmt(calibration_summary.get("best_median_offset"))],
        ["best_signal_count", fmt(calibration_summary.get("best_signal_count") or calibration_summary.get("best_signals_total"))],
    ]
    lines.append(md_table(["Metric", "Value"], calib_rows))
    lines.append("")

    if not sweep_df.empty:
        sw = sweep_df.copy()
        if "score" in sw.columns:
            sw = sw.sort_values("score", ascending=False)
        top_cols = [c for c in [
            "threshold", "min_pending_bars", "drop_delta", "score", "hit0_rate", "hit1_rate",
            "early_rate", "late_rate", "miss_rate", "fp_b_rate", "avg_offset", "median_offset", "signal_count", "signals_total"
        ] if c in sw.columns]
        if top_cols:
            rows: list[list[Any]] = []
            for _, row in sw[top_cols].head(10).iterrows():
                display_row = []
                for col in top_cols:
                    if col.endswith("_rate"):
                        display_row.append(fmt_pct(row[col]))
                    else:
                        display_row.append(fmt(row[col]))
                rows.append(display_row)
            lines.append("### Top calibration rows")
            lines.append("")
            lines.append(md_table(top_cols, rows))
            lines.append("")

    lines.append("## Validation metrics")
    lines.append("")
    if val_summary:
        val_rows = [
            ["signals", fmt(val_summary.get("signals"))],
            ["hit0_rate", fmt_pct(val_summary.get("hit0_rate"))],
            ["hit0_or_hit1_rate", fmt_pct(val_summary.get("hit0_or_hit1_rate"))],
            ["early_rate", fmt_pct(val_summary.get("early_rate"))],
            ["late_rate", fmt_pct(val_summary.get("late_rate"))],
            ["miss_rate", fmt_pct(val_summary.get("miss_rate"))],
            ["fp_b_rate", fmt_pct(val_summary.get("fp_b_rate"))],
            ["pr_auc", fmt(val_summary.get("pr_auc"))],
            ["roc_auc", fmt(val_summary.get("roc_auc"))],
            ["squeeze_median_h32", fmt(val_summary.get("squeeze_median_h32"))],
            ["squeeze_p75_h32", fmt(val_summary.get("squeeze_p75_h32"))],
            ["pullback_median_h32", fmt(val_summary.get("pullback_median_h32"))],
            ["pullback_p75_h32", fmt(val_summary.get("pullback_p75_h32"))],
            ["net_edge_median_h32", fmt(val_summary.get("net_edge_median_h32"))],
            ["clean_2_3_count_h32", fmt(val_summary.get("clean_2_3_count_h32"))],
            ["clean_2_3_share_h32", fmt_pct(val_summary.get("clean_2_3_share_h32"))],
            ["dirty_retrace_2_3_count_h32", fmt(val_summary.get("dirty_retrace_2_3_count_h32"))],
            ["dirty_retrace_2_3_share_h32", fmt_pct(val_summary.get("dirty_retrace_2_3_share_h32"))],
            ["clean_no_pullback_2_3_count_h32", fmt(val_summary.get("clean_no_pullback_2_3_count_h32"))],
            ["clean_no_pullback_2_3_share_h32", fmt_pct(val_summary.get("clean_no_pullback_2_3_share_h32"))],
            ["dirty_no_pullback_2_3_count_h32", fmt(val_summary.get("dirty_no_pullback_2_3_count_h32"))],
            ["dirty_no_pullback_2_3_share_h32", fmt_pct(val_summary.get("dirty_no_pullback_2_3_share_h32"))],
            ["clean_to_dirty_failure_ratio_2_3_h32", fmt(val_summary.get("clean_to_dirty_failure_ratio_2_3_h32"))],
            ["clean_retrace_precision_2_3_h32", fmt(val_summary.get("clean_retrace_precision_2_3_h32"))],
            ["low_squeeze_conversion_2_3_h32", fmt(val_summary.get("low_squeeze_conversion_2_3_h32"))],
            ["pullback_before_squeeze_share_2_3_h32", fmt_pct(val_summary.get("pullback_before_squeeze_share_2_3_h32"))],
            ["trade_quality_score", fmt(val_summary.get("trade_quality_score"))],
        ]
        lines.append(md_table(["Metric", "Value"], val_rows))
        lines.append("")
    else:
        lines.append("Validation artifacts not found.")
        lines.append("")

    lines.append("## Holdout metrics")
    lines.append("")
    if holdout_summary:
        hold_rows = [
            ["signals", fmt(holdout_summary.get("signals"))],
            ["hit0_rate", fmt_pct(holdout_summary.get("hit0_rate"))],
            ["hit0_or_hit1_rate", fmt_pct(holdout_summary.get("hit0_or_hit1_rate"))],
            ["early_rate", fmt_pct(holdout_summary.get("early_rate"))],
            ["late_rate", fmt_pct(holdout_summary.get("late_rate"))],
            ["miss_rate", fmt_pct(holdout_summary.get("miss_rate"))],
            ["fp_b_rate", fmt_pct(holdout_summary.get("fp_b_rate"))],
            ["pr_auc", fmt(holdout_summary.get("pr_auc"))],
            ["roc_auc", fmt(holdout_summary.get("roc_auc"))],
            ["squeeze_median_h32", fmt(holdout_summary.get("squeeze_median_h32"))],
            ["squeeze_p75_h32", fmt(holdout_summary.get("squeeze_p75_h32"))],
            ["pullback_median_h32", fmt(holdout_summary.get("pullback_median_h32"))],
            ["pullback_p75_h32", fmt(holdout_summary.get("pullback_p75_h32"))],
            ["net_edge_median_h32", fmt(holdout_summary.get("net_edge_median_h32"))],
            ["clean_2_3_count_h32", fmt(holdout_summary.get("clean_2_3_count_h32"))],
            ["clean_2_3_share_h32", fmt_pct(holdout_summary.get("clean_2_3_share_h32"))],
            ["dirty_retrace_2_3_count_h32", fmt(holdout_summary.get("dirty_retrace_2_3_count_h32"))],
            ["dirty_retrace_2_3_share_h32", fmt_pct(holdout_summary.get("dirty_retrace_2_3_share_h32"))],
            ["clean_no_pullback_2_3_count_h32", fmt(holdout_summary.get("clean_no_pullback_2_3_count_h32"))],
            ["clean_no_pullback_2_3_share_h32", fmt_pct(holdout_summary.get("clean_no_pullback_2_3_share_h32"))],
            ["dirty_no_pullback_2_3_count_h32", fmt(holdout_summary.get("dirty_no_pullback_2_3_count_h32"))],
            ["dirty_no_pullback_2_3_share_h32", fmt_pct(holdout_summary.get("dirty_no_pullback_2_3_share_h32"))],
            ["clean_to_dirty_failure_ratio_2_3_h32", fmt(holdout_summary.get("clean_to_dirty_failure_ratio_2_3_h32"))],
            ["clean_retrace_precision_2_3_h32", fmt(holdout_summary.get("clean_retrace_precision_2_3_h32"))],
            ["low_squeeze_conversion_2_3_h32", fmt(holdout_summary.get("low_squeeze_conversion_2_3_h32"))],
            ["pullback_before_squeeze_share_2_3_h32", fmt_pct(holdout_summary.get("pullback_before_squeeze_share_2_3_h32"))],
            ["signals_per_30d", fmt(holdout_summary.get("signals_per_30d"))],
            ["symbols", fmt(holdout_summary.get("symbols"))],
            ["signal_offset_p50", fmt(holdout_summary.get("signal_offset_p50") or holdout_summary.get("median_pred_offset"))],
            ["signal_offset_p90", fmt(holdout_summary.get("signal_offset_p90"))],
            ["tp", fmt(holdout_summary.get("tp"))],
            ["sl", fmt(holdout_summary.get("sl"))],
            ["timeout", fmt(holdout_summary.get("timeout"))],
            ["ambiguous", fmt(holdout_summary.get("ambiguous"))],
            ["tp_rate_resolved", fmt_pct(holdout_summary.get("tp_rate_resolved"))],
            ["sl_rate_resolved", fmt_pct(holdout_summary.get("sl_rate_resolved"))],
            ["pnl_sum", fmt(holdout_summary.get("pnl_sum"))],
            ["expectancy_all", fmt(holdout_summary.get("expectancy_all"))],
            ["expectancy_resolved", fmt(holdout_summary.get("expectancy_resolved"))],
            ["profit_factor", fmt(holdout_summary.get("profit_factor"))],
            ["max_losing_streak", fmt(holdout_summary.get("max_losing_streak"))],
            ["worst_6h_pnl", fmt(holdout_summary.get("worst_6h_pnl"))],
            ["worst_24h_pnl", fmt(holdout_summary.get("worst_24h_pnl"))],
            ["trade_quality_score", fmt(holdout_summary.get("trade_quality_score"))],
            ["mfe_short_32_median", fmt(holdout_summary.get("mfe_short_32_median"))],
            ["mfe_short_32_pct_above_2pct", fmt_pct(holdout_summary.get("mfe_short_32_pct_above_2pct"))],
            ["mae_short_32_median", fmt(holdout_summary.get("mae_short_32_median"))],
        ]
        lines.append(md_table(["Metric", "Value"], hold_rows))
        lines.append("")
    else:
        lines.append("Holdout artifacts not found.")
        lines.append("")

    if not monthly_df.empty:
        cols = [c for c in ["month", "signals", "tp", "sl", "timeout", "pnl_sum", "expectancy"] if c in monthly_df.columns]
        rows = []
        for _, row in monthly_df[cols].iterrows():
            rows.append([fmt(row[c]) for c in cols])
        lines.append("### Holdout by month")
        lines.append("")
        lines.append(md_table(cols, rows))
        lines.append("")

    if not worst_6h_df.empty:
        cols = [c for c in ["bucket_6h", "signals", "tp", "sl", "timeout", "pnl_sum", "expectancy"] if c in worst_6h_df.columns]
        rows = []
        for _, row in worst_6h_df[cols].head(10).iterrows():
            rows.append([fmt(row[c]) for c in cols])
        lines.append("### Worst 6h windows")
        lines.append("")
        lines.append(md_table(cols, rows))
        lines.append("")

    if not worst_symbols_df.empty:
        cols = [c for c in ["symbol", "signals", "sl", "pnl_sum", "expectancy"] if c in worst_symbols_df.columns]
        rows = []
        for _, row in worst_symbols_df[cols].head(12).iterrows():
            rows.append([fmt(row[c]) for c in cols])
        lines.append("### Worst symbols on holdout")
        lines.append("")
        lines.append(md_table(cols, rows))
        lines.append("")

    lines.append("## Event-centric diagnostics")
    lines.append("")
    if eventcentric_summary:
        event_rows = [
            ["signals", fmt(eventcentric_summary.get("signals"))],
            ["hit0_rate", fmt_pct(eventcentric_summary.get("hit0_rate"))],
            ["hit0_or_hit1_rate", fmt_pct(eventcentric_summary.get("hit0_or_hit1_rate"))],
            ["early_rate", fmt_pct(eventcentric_summary.get("early_rate"))],
            ["late_rate", fmt_pct(eventcentric_summary.get("late_rate"))],
            ["miss_rate", fmt_pct(eventcentric_summary.get("miss_rate"))],
            ["fp_b_rate", fmt_pct(eventcentric_summary.get("fp_b_rate"))],
            ["pr_auc", fmt(eventcentric_summary.get("pr_auc"))],
            ["roc_auc", fmt(eventcentric_summary.get("roc_auc"))],
        ]
        lines.append(md_table(["Metric", "Value"], event_rows))
        lines.append("")
    else:
        lines.append("Event-centric artifacts not found.")
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
        lines.append(md_table(["Metric", "Value"], fi_rows))
        lines.append("")

        if not feature_grouped.empty:
            fg = feature_grouped.copy()
            cols = [c for c in ["feature_group", "group", "sum_importance", "mean_importance", "count", "feature_count"] if c in fg.columns]
            if cols:
                rows = []
                for _, row in fg[cols].head(12).iterrows():
                    rows.append([fmt(row[c]) if c not in {"feature_group", "group"} else row[c] for c in cols])
                lines.append("### Feature groups")
                lines.append("")
                lines.append(md_table(cols, rows))
                lines.append("")

        if not feature_importance.empty:
            top_cols = [c for c in ["feature", "importance"] if c in feature_importance.columns]
            if top_cols:
                rows = []
                for _, row in feature_importance[top_cols].head(20).iterrows():
                    rows.append([row["feature"], fmt(row["importance"]), feature_group(str(row["feature"]))])
                lines.append("### Top features")
                lines.append("")
                lines.append(md_table(["Feature", "Importance", "Group"], rows))
                lines.append("")
    else:
        lines.append("Feature importance artifacts not found.")
        lines.append("")

    lines.append("## Artifact inventory")
    lines.append("")
    file_rows: list[list[Any]] = []
    for name in KEY_FILES:
        path = run_dir / name
        if path.exists():
            file_rows.append([name, "yes", fmt(path.stat().st_size)])
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
