import argparse
import json
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
    if "val" in split_targets:
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
        val_signals = extract_signals_from_probability_stream(
            probability_stream_df=val_stream,
            threshold=threshold,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            abstain_margin=abstain_margin,
            eval_start=train_end,
            eval_end=val_end,
        )
        val_signals = finalize_live_shadow_signals(
            signals_df=val_signals,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
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
        artifacts.save_metrics(val_metrics, "holdout_live_shadow_val")
        artifacts.save_predicted_signals_holdout_live_shadow_val(val_signals)

    if "test" in split_targets:
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
        test_signals = extract_signals_from_probability_stream(
            probability_stream_df=test_stream,
            threshold=threshold,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            abstain_margin=abstain_margin,
            eval_start=val_end,
            eval_end=test_end,
        )
        test_signals = finalize_live_shadow_signals(
            signals_df=test_signals,
            loader=loader,
            quality_entry_shift_bars=quality_entry_shift_bars,
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
        artifacts.save_shadow_probability_stream(test_stream, "test")
        artifacts.save_predicted_signals_holdout_live_shadow(test_signals)
        artifacts.save_metrics(test_metrics, "holdout_live_shadow")
        artifacts.save_holdout_window_summary_6h(build_holdout_window_summary_6h(test_signals))
        artifacts.save_holdout_symbol_summary(build_holdout_symbol_summary(test_signals))


if __name__ == "__main__":
    main()
