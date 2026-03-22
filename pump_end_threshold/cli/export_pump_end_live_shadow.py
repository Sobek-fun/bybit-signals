import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pump_end_threshold.cli.train_pump_end_model import (
    build_holdout_symbol_summary,
    build_holdout_window_summary_6h,
    build_live_shadow_metrics,
    extract_signals_from_probability_stream,
    finalize_live_shadow_signals,
    parse_date_exclusive,
    resolve_universe_tokens,
)
from pump_end_threshold.infra.clickhouse import DataLoader
from pump_end_threshold.ml.artifacts import RunArtifacts
from pump_end_threshold.ml.export_signals import export_probability_stream


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}", flush=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_tokens(
        run_dir: Path,
        run_config: dict,
        symbols_file: str | None,
        symbols_csv: str | None,
        dataset_parquet: str | None,
) -> list[str]:
    if symbols_file:
        symbols = [x.strip().upper() for x in Path(symbols_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        return sorted(set([s[:-4] if s.endswith("USDT") else s for s in symbols]))
    if symbols_csv:
        symbols = [x.strip().upper() for x in symbols_csv.split(",") if x.strip()]
        return sorted(set([s[:-4] if s.endswith("USDT") else s for s in symbols]))

    candidate_paths = []
    if dataset_parquet:
        candidate_paths.append(Path(dataset_parquet))
    if run_config.get("dataset_parquet"):
        candidate_paths.append(Path(str(run_config["dataset_parquet"])))
    candidate_paths.append(run_dir / "features.parquet")
    for p in candidate_paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["symbol"])
        return resolve_universe_tokens(df)
    raise ValueError("unable to resolve symbols/tokens: provide --symbols-file/--symbols-csv or dataset parquet with symbol column")


def _split_bounds(run_config: dict) -> tuple[datetime, datetime, datetime]:
    train_end_raw = run_config.get("train_end")
    val_end_raw = run_config.get("val_end")
    test_end_raw = run_config.get("test_end")
    if not train_end_raw or not val_end_raw or not test_end_raw:
        raise ValueError("run_config.json must contain train_end, val_end, test_end")
    train_end = parse_date_exclusive(str(train_end_raw))
    val_end = parse_date_exclusive(str(val_end_raw))
    test_end = parse_date_exclusive(str(test_end_raw))
    return train_end, val_end, test_end


def _count_unique_symbols(df: pd.DataFrame) -> int:
    if df.empty or "symbol" not in df.columns:
        return 0
    return int(df["symbol"].nunique())


def _extract_trade_outcome_stats(df: pd.DataFrame) -> dict:
    stats = {
        "tp": 0,
        "sl": 0,
        "timeout": 0,
        "ambiguous": 0,
    }
    if not df.empty and "trade_outcome" in df.columns:
        outcomes = df["trade_outcome"].astype(str).str.lower()
        stats["tp"] = int((outcomes == "tp").sum())
        stats["sl"] = int((outcomes == "sl").sum())
        stats["timeout"] = int((outcomes == "timeout").sum())
    if not df.empty:
        if "trade_replay_ambiguous" in df.columns:
            stats["ambiguous"] = int(pd.to_numeric(df["trade_replay_ambiguous"], errors="coerce").fillna(0).astype(int).sum())
        elif "ambiguous" in df.columns:
            stats["ambiguous"] = int(pd.to_numeric(df["ambiguous"], errors="coerce").fillna(0).astype(int).sum())
        elif "is_ambiguous" in df.columns:
            stats["ambiguous"] = int(pd.to_numeric(df["is_ambiguous"], errors="coerce").fillna(0).astype(int).sum())
        elif "ambiguous_bar_time" in df.columns:
            stats["ambiguous"] = int(df["ambiguous_bar_time"].notna().sum())
    return stats


def _count_stream_parts(run_dir: Path, split: str) -> int:
    pattern = f"shadow_probability_stream_{split}_part*.parquet"
    return sum(1 for _ in run_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Export live-shadow outputs from existing run artifacts")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--clickhouse-dsn", type=str, required=True)
    parser.add_argument("--symbols-file", type=str, default=None)
    parser.add_argument("--symbols-csv", type=str, default=None)
    parser.add_argument("--dataset-parquet", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test", "both"], default="test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stream-fast-mode", action="store_true", default=False)
    args = parser.parse_args()
    script_started = time.perf_counter()
    log("INFO", "EXPORT_LIVE_SHADOW", f"start run_dir={args.run_dir} split={args.split} workers={args.workers}")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    run_config = _load_json(run_dir / "run_config.json")
    best_threshold = _load_json(run_dir / "best_threshold.json")
    if not best_threshold:
        raise SystemExit(f"best_threshold.json not found in {run_dir}")
    if not (run_dir / "catboost_model.cbm").exists():
        raise SystemExit(f"catboost_model.cbm not found in {run_dir}")

    train_end, val_end, test_end = _split_bounds(run_config)
    tokens = _resolve_tokens(
        run_dir=run_dir,
        run_config=run_config,
        symbols_file=args.symbols_file,
        symbols_csv=args.symbols_csv,
        dataset_parquet=args.dataset_parquet,
    )
    artifacts = RunArtifacts(str(run_dir.parent), run_dir.name)
    loader = DataLoader(args.clickhouse_dsn)
    threshold = float(best_threshold.get("threshold", 0.1))
    min_pending_bars = int(best_threshold.get("min_pending_bars", 1))
    drop_delta = float(best_threshold.get("drop_delta", 0.0))
    abstain_margin = float(best_threshold.get("abstain_margin", run_config.get("abstain_margin", 0.0)))
    quality_entry_shift_bars = int(run_config.get("quality_entry_shift_bars", 0))
    quality_density_mode = str(run_config.get("quality_density_mode", "per30d"))
    quality_target_min_30d = float(run_config.get("quality_target_min_30d", 35.0))
    quality_target_max_30d = float(run_config.get("quality_target_max_30d", 75.0))
    quality_overflow_penalty = float(run_config.get("quality_overflow_penalty", 0.08))

    split_targets = ["val", "test"] if args.split == "both" else [args.split]
    log("INFO", "EXPORT_LIVE_SHADOW", f"resolved split_targets={','.join(split_targets)} tokens={len(tokens)}")
    if "val" in split_targets:
        val_export_started = time.perf_counter()
        log("INFO", "EXPORT_STREAM", f"split=val start tokens={len(tokens)} dt_from={train_end} dt_to={val_end - timedelta(minutes=15)}")
        val_stream = export_probability_stream(
            tokens=tokens,
            ch_dsn=args.clickhouse_dsn,
            model_dir=str(run_dir),
            dt_from=train_end,
            dt_to=val_end - timedelta(minutes=15),
            workers=max(1, int(args.workers)),
            out_parquet=str(run_dir / "shadow_probability_stream_val.parquet"),
            stream_fast_mode=args.stream_fast_mode,
        )
        log(
            "INFO",
            "EXPORT_STREAM",
            f"split=val done rows={len(val_stream)} symbols={_count_unique_symbols(val_stream)} elapsed_sec={time.perf_counter() - val_export_started:.2f}",
        )
        val_extract_started = time.perf_counter()
        log(
            "INFO",
            "SHADOW_EXTRACT",
            f"split=val start part_files={_count_stream_parts(run_dir, 'val')} threshold={threshold} min_pending_bars={min_pending_bars} drop_delta={drop_delta} abstain_margin={abstain_margin}",
        )
        val_signals = extract_signals_from_probability_stream(
            probability_stream_df=val_stream,
            threshold=threshold,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            abstain_margin=abstain_margin,
            eval_start=train_end,
            eval_end=val_end,
        )
        log(
            "INFO",
            "SHADOW_EXTRACT",
            f"split=val done signals={len(val_signals)} symbols={_count_unique_symbols(val_signals)} elapsed_sec={time.perf_counter() - val_extract_started:.2f}",
        )
        val_finalize_started = time.perf_counter()
        log(
            "INFO",
            "SHADOW_QUALITY",
            f"split=val start stage=finalize_live_shadow_signals signals={len(val_signals)} symbols={_count_unique_symbols(val_signals)} quality_entry_shift_bars={quality_entry_shift_bars}",
        )
        log(
            "INFO",
            "SHADOW_REPLAY",
            f"split=val start stage=finalize_live_shadow_signals signals={len(val_signals)}",
        )
        val_signals = finalize_live_shadow_signals(
            signals_df=val_signals,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
        )
        log(
            "INFO",
            "SHADOW_QUALITY",
            f"split=val done stage=finalize_live_shadow_signals rows={len(val_signals)} elapsed_sec={time.perf_counter() - val_finalize_started:.2f}",
        )
        val_outcome_stats = _extract_trade_outcome_stats(val_signals)
        log(
            "INFO",
            "SHADOW_REPLAY",
            f"split=val done stage=finalize_live_shadow_signals rows={len(val_signals)} ambiguous={val_outcome_stats['ambiguous']} tp={val_outcome_stats['tp']} sl={val_outcome_stats['sl']} timeout={val_outcome_stats['timeout']} elapsed_sec={time.perf_counter() - val_finalize_started:.2f}",
        )
        val_metrics_started = time.perf_counter()
        has_quality_col_val = int("squeeze_pct_h32" in val_signals.columns)
        log(
            "INFO",
            "SHADOW_METRICS",
            f"split=val start signals={len(val_signals)} has_squeeze_pct_h32={has_quality_col_val}",
        )
        val_metrics = build_live_shadow_metrics(
            signals_df=val_signals,
            window_start=train_end,
            window_end=val_end,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
            quality_density_mode=quality_density_mode,
            quality_target_min_30d=quality_target_min_30d,
            quality_target_max_30d=quality_target_max_30d,
            quality_overflow_penalty=quality_overflow_penalty,
        )
        log(
            "INFO",
            "SHADOW_METRICS",
            f"split=val done trade_quality_score={val_metrics.get('trade_quality_score')} pnl_sum={val_metrics.get('pnl_sum')} worst_6h_pnl={val_metrics.get('worst_6h_pnl')} worst_24h_pnl={val_metrics.get('worst_24h_pnl')} elapsed_sec={time.perf_counter() - val_metrics_started:.2f}",
        )
        val_save_started = time.perf_counter()
        log("INFO", "SHADOW_SAVE", "split=val start file=metrics_holdout_live_shadow_val.json")
        artifacts.save_metrics(val_metrics, "holdout_live_shadow_val")
        log("INFO", "SHADOW_SAVE", f"split=val done file=metrics_holdout_live_shadow_val.json rows=1 elapsed_sec={time.perf_counter() - val_save_started:.2f}")
        val_signals_save_started = time.perf_counter()
        log("INFO", "SHADOW_SAVE", f"split=val start file=predicted_signals_holdout_live_shadow_val.csv rows={len(val_signals)}")
        artifacts.save_predicted_signals_holdout_live_shadow_val(val_signals)
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=val done file=predicted_signals_holdout_live_shadow_val.csv rows={len(val_signals)} elapsed_sec={time.perf_counter() - val_signals_save_started:.2f}",
        )

    if "test" in split_targets:
        test_export_started = time.perf_counter()
        log("INFO", "EXPORT_STREAM", f"split=test start tokens={len(tokens)} dt_from={val_end} dt_to={test_end - timedelta(minutes=15)}")
        test_stream = export_probability_stream(
            tokens=tokens,
            ch_dsn=args.clickhouse_dsn,
            model_dir=str(run_dir),
            dt_from=val_end,
            dt_to=test_end - timedelta(minutes=15),
            workers=max(1, int(args.workers)),
            out_parquet=str(run_dir / "shadow_probability_stream_test.parquet"),
            stream_fast_mode=args.stream_fast_mode,
        )
        log(
            "INFO",
            "EXPORT_STREAM",
            f"split=test done rows={len(test_stream)} symbols={_count_unique_symbols(test_stream)} elapsed_sec={time.perf_counter() - test_export_started:.2f}",
        )
        test_extract_started = time.perf_counter()
        log(
            "INFO",
            "SHADOW_EXTRACT",
            f"split=test start part_files={_count_stream_parts(run_dir, 'test')} threshold={threshold} min_pending_bars={min_pending_bars} drop_delta={drop_delta} abstain_margin={abstain_margin}",
        )
        test_signals = extract_signals_from_probability_stream(
            probability_stream_df=test_stream,
            threshold=threshold,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            abstain_margin=abstain_margin,
            eval_start=val_end,
            eval_end=test_end,
        )
        log(
            "INFO",
            "SHADOW_EXTRACT",
            f"split=test done signals={len(test_signals)} symbols={_count_unique_symbols(test_signals)} elapsed_sec={time.perf_counter() - test_extract_started:.2f}",
        )
        test_finalize_started = time.perf_counter()
        log(
            "INFO",
            "SHADOW_QUALITY",
            f"split=test start stage=finalize_live_shadow_signals signals={len(test_signals)} symbols={_count_unique_symbols(test_signals)} quality_entry_shift_bars={quality_entry_shift_bars}",
        )
        log(
            "INFO",
            "SHADOW_REPLAY",
            f"split=test start stage=finalize_live_shadow_signals signals={len(test_signals)}",
        )
        test_signals = finalize_live_shadow_signals(
            signals_df=test_signals,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
        )
        log(
            "INFO",
            "SHADOW_QUALITY",
            f"split=test done stage=finalize_live_shadow_signals rows={len(test_signals)} elapsed_sec={time.perf_counter() - test_finalize_started:.2f}",
        )
        test_outcome_stats = _extract_trade_outcome_stats(test_signals)
        log(
            "INFO",
            "SHADOW_REPLAY",
            f"split=test done stage=finalize_live_shadow_signals rows={len(test_signals)} ambiguous={test_outcome_stats['ambiguous']} tp={test_outcome_stats['tp']} sl={test_outcome_stats['sl']} timeout={test_outcome_stats['timeout']} elapsed_sec={time.perf_counter() - test_finalize_started:.2f}",
        )
        test_metrics_started = time.perf_counter()
        has_quality_col_test = int("squeeze_pct_h32" in test_signals.columns)
        log(
            "INFO",
            "SHADOW_METRICS",
            f"split=test start signals={len(test_signals)} has_squeeze_pct_h32={has_quality_col_test}",
        )
        test_metrics = build_live_shadow_metrics(
            signals_df=test_signals,
            window_start=val_end,
            window_end=test_end,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
            quality_density_mode=quality_density_mode,
            quality_target_min_30d=quality_target_min_30d,
            quality_target_max_30d=quality_target_max_30d,
            quality_overflow_penalty=quality_overflow_penalty,
        )
        log(
            "INFO",
            "SHADOW_METRICS",
            f"split=test done trade_quality_score={test_metrics.get('trade_quality_score')} pnl_sum={test_metrics.get('pnl_sum')} worst_6h_pnl={test_metrics.get('worst_6h_pnl')} worst_24h_pnl={test_metrics.get('worst_24h_pnl')} elapsed_sec={time.perf_counter() - test_metrics_started:.2f}",
        )
        test_save_started = time.perf_counter()
        log("INFO", "SHADOW_SAVE", f"split=test start file=shadow_probability_stream_test.parquet rows={len(test_stream)}")
        artifacts.save_shadow_probability_stream(test_stream, "test")
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=test done file=shadow_probability_stream_test.parquet rows={len(test_stream)} elapsed_sec={time.perf_counter() - test_save_started:.2f}",
        )
        test_signals_save_started = time.perf_counter()
        log("INFO", "SHADOW_SAVE", f"split=test start file=predicted_signals_holdout_live_shadow.csv rows={len(test_signals)}")
        artifacts.save_predicted_signals_holdout_live_shadow(test_signals)
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=test done file=predicted_signals_holdout_live_shadow.csv rows={len(test_signals)} elapsed_sec={time.perf_counter() - test_signals_save_started:.2f}",
        )
        test_metrics_save_started = time.perf_counter()
        log("INFO", "SHADOW_SAVE", "split=test start file=metrics_holdout_live_shadow.json")
        artifacts.save_metrics(test_metrics, "holdout_live_shadow")
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=test done file=metrics_holdout_live_shadow.json rows=1 elapsed_sec={time.perf_counter() - test_metrics_save_started:.2f}",
        )
        test_windows_save_started = time.perf_counter()
        holdout_windows_6h = build_holdout_window_summary_6h(test_signals)
        log("INFO", "SHADOW_SAVE", f"split=test start file=holdout_window_summary_6h.csv rows={len(holdout_windows_6h)}")
        artifacts.save_holdout_window_summary_6h(holdout_windows_6h)
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=test done file=holdout_window_summary_6h.csv rows={len(holdout_windows_6h)} elapsed_sec={time.perf_counter() - test_windows_save_started:.2f}",
        )
        test_symbols_save_started = time.perf_counter()
        holdout_symbol_summary = build_holdout_symbol_summary(test_signals)
        log("INFO", "SHADOW_SAVE", f"split=test start file=holdout_symbol_summary.csv rows={len(holdout_symbol_summary)}")
        artifacts.save_holdout_symbol_summary(holdout_symbol_summary)
        log(
            "INFO",
            "SHADOW_SAVE",
            f"split=test done file=holdout_symbol_summary.csv rows={len(holdout_symbol_summary)} elapsed_sec={time.perf_counter() - test_symbols_save_started:.2f}",
        )
    log("INFO", "EXPORT_LIVE_SHADOW", f"done elapsed_sec={time.perf_counter() - script_started:.2f}")


if __name__ == "__main__":
    main()
