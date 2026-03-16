import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
from pump_end_threshold.infra.clickhouse import DataLoader, get_liquid_universe
from pump_end_threshold.ml.regime_dataset import (
    build_strategy_state,
    build_strategy_state_live,
    BAR_MINUTES,
    ENTRY_SHIFT_BARS,
    STRATEGY_STATE_MODE,
)


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def load_symbols_from_file(path: str) -> list[str]:
    lines = Path(path).read_text().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def validate_signals_source(signals_path: str, signals_df: pd.DataFrame) -> dict:
    path = Path(signals_path)
    file_name = path.name.lower()
    is_cv_oos = file_name == "cv_oos_signals_verbose.parquet"

    if is_cv_oos:
        source_col_oos = False
        if 'source_model_run' in signals_df.columns:
            source_model_run = signals_df['source_model_run'].astype(str).str.lower()
            source_col_oos = bool(source_model_run.str.contains('oos').all())

        flag_col_oos = False
        if 'base_model_oos' in signals_df.columns:
            base_model_oos = pd.to_numeric(signals_df['base_model_oos'], errors='coerce').fillna(0)
            flag_col_oos = bool((base_model_oos > 0).all())

        manifest_oos = False
        for candidate in [
            path.with_suffix(path.suffix + ".manifest.json"),
            path.parent / "dataset_manifest.json",
            path.parent / "signals_manifest.json",
            path.parent / "manifest.json",
        ]:
            if not candidate.exists():
                continue
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                manifest_oos = bool(
                    payload.get("base_model_oos")
                    or payload.get("is_oos")
                    or payload.get("oos")
                    or payload.get("base_model_is_oos")
                )
                if manifest_oos:
                    break
            except Exception:
                continue

        validated = bool(source_col_oos or flag_col_oos or manifest_oos)
        if not validated:
            raise ValueError(
                "cv_oos_signals_verbose.parquet должен содержать подтверждение OOS provenance "
                "(source_model_run~oos, base_model_oos=true или manifest с OOS=true)"
            )
        return {
            'signals_source_type': 'cv_oos_base_model',
            'base_model_oos_validated': True,
        }

    has_required = {'symbol', 'open_time'}.issubset(signals_df.columns)
    detector_hint_cols = {
        'event_type',
        'event_id',
        'signal_offset',
        'p_end_at_fire',
        'threshold_used',
        'drop_from_peak_at_fire',
    }
    has_detector_hints = any(c in signals_df.columns for c in detector_hint_cols)

    strict_forbidden_cols = {
        'blocked_by_policy',
        'accepted_by_policy',
        'regime_state',
        'bucket_p_bad',
    }
    has_strict_forbidden = any(c in signals_df.columns for c in strict_forbidden_cols)
    has_target_like_cols = any(
        col.startswith(('target_', 'future_'))
        for col in signals_df.columns
    )
    has_trade_label_cols = any(
        col in signals_df.columns
        for col in ('trade_outcome', 'pnl_pct', 'tp_hit', 'sl_hit')
    )
    has_regime_score = 'p_bad' in signals_df.columns

    if (
            not has_required or
            not has_detector_hints or
            has_strict_forbidden or
            has_target_like_cols or
            has_trade_label_cols or
            has_regime_score
    ):
        raise ValueError(
            "Для regime разрешены только два источника сигналов: raw stage-A detector export "
            "или cv_oos_signals_verbose.parquet"
        )

    return {
        'signals_source_type': 'raw_stage_a_detector',
        'base_model_oos_validated': True,
    }


def build_strategy_state_live_by_time(
        signals_df: pd.DataFrame,
        loader: DataLoader,
        tp_pct: float,
        sl_pct: float,
        max_horizon_bars: int,
        trade_replay_source: str,
) -> dict:
    state_by_time = {}
    unique_times = pd.Series(signals_df['open_time'].dropna().unique()).sort_values()
    for t in unique_times:
        asof_time = pd.Timestamp(t).to_pydatetime()
        state_by_time[pd.Timestamp(t)] = build_strategy_state_live(
            signals_df,
            loader,
            asof_time=asof_time,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_horizon_bars=max_horizon_bars,
            trade_replay_source=trade_replay_source,
        )
    return state_by_time


def main():
    parser = argparse.ArgumentParser(description="Build regime guard dataset from detector signals")
    parser.add_argument("--signals-path", type=str, required=True,
                        help="Path to detector signals (CSV or Parquet with verbose detector metadata)")
    parser.add_argument("--clickhouse-dsn", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directory for all artifacts (if not provided, uses --output)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: RUN_DIR/regime_dataset.parquet)")
    parser.add_argument("--top-n-universe", type=int, default=120)
    parser.add_argument("--tp-pct", type=float, default=4.5)
    parser.add_argument("--sl-pct", type=float, default=10.0)
    parser.add_argument("--max-horizon-bars", type=int, default=200)
    parser.add_argument("--target-horizon-signals", type=int, default=5)
    parser.add_argument("--target-min-resolved", type=int, default=3)
    parser.add_argument("--target-sl-rate-threshold", type=float, default=0.60)
    parser.add_argument("--target-col", type=str, default="target_pause_value_next_12h",
                        help="Name of the target column to use for training")
    parser.add_argument("--include-detector-snapshot", action="store_true",
                        help="Include detector snapshot features from PumpFeatureBuilder")
    parser.add_argument("--save-debug-sample", action="store_true",
                        help="Save debug sample of dataset")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="Path to file with allowed symbols (one per line)")
    parser.add_argument("--fixed-universe-file", type=str, default=None,
                        help="Path to file with fixed universe symbols (one per line), skips liquid universe computation")
    parser.add_argument("--trade-replay-source", type=str, default="1s",
                        choices=["1m", "1s"],
                        help="Trade replay source: 1m (fast, no 1s resolve) or 1s (exact with 1s replay)")
    parser.add_argument("--target-profile", type=str, default=None,
                        help="Target profile name (e.g., pause_value_12h_v2_all, pause_value_12h_v2_curated)")
    parser.add_argument("--feature-profile", type=str, default=None,
                        help="Feature profile name for documentation/versioning")

    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.output:
            args.output = str(run_dir / "regime_dataset.parquet")
    else:
        run_dir = None
        if not args.output:
            raise ValueError("Either --run-dir or --output must be specified")

    log("INFO", "REGIME-DS", f"loading signals from {args.signals_path}")
    if args.signals_path.endswith('.parquet'):
        signals_df = pd.read_parquet(args.signals_path)
    else:
        signals_df = pd.read_csv(args.signals_path)

    if 'timestamp' in signals_df.columns and 'open_time' not in signals_df.columns:
        signals_df = signals_df.rename(columns={'timestamp': 'open_time'})
    signals_df['open_time'] = pd.to_datetime(signals_df['open_time'])
    signals_df = signals_df.sort_values('open_time').reset_index(drop=True)
    source_meta = validate_signals_source(args.signals_path, signals_df)

    if args.symbols_file:
        allowed_symbols = set(load_symbols_from_file(args.symbols_file))
        before = len(signals_df)
        signals_df = signals_df[signals_df['symbol'].isin(allowed_symbols)].reset_index(drop=True)
        log("INFO", "REGIME-DS", f"filtered by symbols-file: {before} -> {len(signals_df)}")

    if 'signal_id' not in signals_df.columns:
        if 'event_id' in signals_df.columns:
            signals_df['signal_id'] = signals_df['event_id']
        else:
            signals_df['signal_id'] = [
                f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}|{row.get('signal_offset', 0)}"
                for _, row in signals_df.iterrows()
            ]

    if 'event_type' not in signals_df.columns:
        signals_df['event_type'] = 'A'

    log("INFO", "REGIME-DS", f"loaded {len(signals_df)} signals")

    loader = DataLoader(args.clickhouse_dsn)

    t_min = signals_df['open_time'].min()

    if args.fixed_universe_file:
        liquid_universe = load_symbols_from_file(args.fixed_universe_file)
        log("INFO", "REGIME-DS", f"using fixed universe: {len(liquid_universe)} symbols")
    else:
        log("INFO", "REGIME-DS", "fetching liquid universe")
        liquid_universe = get_liquid_universe(
            args.clickhouse_dsn, t_min - timedelta(days=7), t_min,
            top_n=args.top_n_universe,
        )
        log("INFO", "REGIME-DS", f"liquid universe: {len(liquid_universe)} symbols")

    if run_dir:
        with open(run_dir / "liquid_universe.json", 'w') as f:
            json.dump(liquid_universe, f, indent=2)
        log("INFO", "REGIME-DS", "saved liquid_universe.json")

    log("INFO", "REGIME-DS", "simulating trades")
    trades_df = build_strategy_state(
        signals_df, loader,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        max_horizon_bars=args.max_horizon_bars,
        trade_replay_source=args.trade_replay_source,
    )
    log("INFO", "REGIME-DS", f"trades simulated: {len(trades_df)}")

    if 'trade_outcome' in trades_df.columns:
        tp_count = (trades_df['trade_outcome'] == 'TP').sum()
        sl_count = (trades_df['trade_outcome'] == 'SL').sum()
        unknown = (trades_df['trade_outcome'] == 'UNKNOWN').sum()
        timeout = (trades_df['trade_outcome'] == 'TIMEOUT').sum()
        ambiguous = (trades_df['trade_outcome'] == 'AMBIGUOUS').sum()
        log("INFO", "REGIME-DS", f"outcomes: TP={tp_count} SL={sl_count} UNKNOWN={unknown} TIMEOUT={timeout} AMBIGUOUS={ambiguous}")

    log("INFO", "REGIME-DS", "building regime features")

    builder = RegimeFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        liquid_universe=liquid_universe,
        top_n=args.top_n_universe,
    )

    if run_dir:
        regime_builder_config = {
            'top_n_universe': args.top_n_universe,
            'feature_version': 'v4',
            'feature_profile': args.feature_profile or 'regime_compact_v4',
            'trade_replay_source': args.trade_replay_source,
            'target_profile': args.target_profile,
            'strategy_state_mode': STRATEGY_STATE_MODE,
        }
        with open(run_dir / "regime_builder_config.json", 'w') as f:
            json.dump(regime_builder_config, f, indent=2)
        log("INFO", "REGIME-DS", "saved regime_builder_config.json")

    log("INFO", "REGIME-DS", "building regime features in batch mode")

    pump_builder = None
    if args.include_detector_snapshot:
        from pump_end_threshold.features.feature_builder import PumpFeatureBuilder
        pump_builder = PumpFeatureBuilder(
            ch_dsn=args.clickhouse_dsn,
            window_bars=30,
            warmup_bars=150,
            feature_set='snapshot'
        )
        log("INFO", "REGIME-DS", "including detector snapshot features")

    batch_size = 50
    log("INFO", "REGIME-DS", "building causal strategy-state snapshots")
    strategy_state_by_time = build_strategy_state_live_by_time(
        signals_df=signals_df,
        loader=loader,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        max_horizon_bars=args.max_horizon_bars,
        trade_replay_source=args.trade_replay_source,
    )
    log("INFO", "REGIME-DS", "building features")
    regime_features = builder.build_batch(signals_df, batch_size=batch_size, trades_df=strategy_state_by_time)

    if regime_features.empty:
        log("ERROR", "REGIME-DS", "regime_features is empty!")
        return

    log("INFO", "REGIME-DS", f"regime_features shape: {regime_features.shape}")
    log("INFO", "REGIME-DS", f"regime_features columns sample: {list(regime_features.columns)[:10]}")

    from pump_end_threshold.ml.regime_dataset import build_regime_dataset as build_dataset_helper

    dataset = build_dataset_helper(
        signals_df=signals_df,
        features_df=regime_features,
        trades_df=trades_df,
        min_resolved=args.target_min_resolved,
        sl_rate_threshold=args.target_sl_rate_threshold,
        target_profile=args.target_profile,
        max_horizon_bars=args.max_horizon_bars,
    )

    if dataset.empty:
        log("ERROR", "REGIME-DS", "dataset from helper is empty!")
        return

    log("INFO", "REGIME-DS", f"dataset after helper shape: {dataset.shape}")
    log("INFO", "REGIME-DS", f"dataset columns sample: {list(dataset.columns)[:20]}")

    if pump_builder and args.include_detector_snapshot:
        log("INFO", "REGIME-DS", "adding detector snapshot features")
        for idx, sig in signals_df.iterrows():
            if idx >= len(dataset):
                break
            sig_batch = signals_df.iloc[max(0, idx-batch_size):idx+1].copy()
            sig_batch = sig_batch.rename(columns={'open_time': 'event_open_time'})
            sig_batch['pump_la_type'] = 'A'
            sig_batch['runup_pct'] = 0
            try:
                pump_feats_batch = pump_builder.build(sig_batch, max_workers=1)
                if not pump_feats_batch.empty and idx < len(pump_feats_batch):
                    pump_row = pump_feats_batch.iloc[idx % len(pump_feats_batch)]
                    for col in pump_row.index:
                        if col not in dataset.columns and col != 'signal_id':
                            dataset.loc[idx, col] = pump_row[col]
            except:
                pass

    log("INFO", "REGIME-DS", f"final dataset shape: {dataset.shape}")
    log("INFO", "REGIME-DS", f"final columns sample: {list(dataset.columns)[:20]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(out_path), index=False)
    log("INFO", "REGIME-DS", f"saved to {out_path}")

    if 'target_pause_value_next_12h' in dataset.columns:
        valid = dataset['target_pause_value_next_12h'].dropna()
        if len(valid) > 0:
            log("INFO", "REGIME-DS", f"target_pause_value_next_12h: positive_rate={valid.mean():.3f}, n_valid={len(valid)}, n_nan={dataset['target_pause_value_next_12h'].isna().sum()}")

    if run_dir:
        config = {
            'signals_path': args.signals_path,
            'n_signals': len(signals_df),
            'n_samples': len(dataset),
            'tp_pct': args.tp_pct,
            'sl_pct': args.sl_pct,
            'max_horizon_bars': args.max_horizon_bars,
            'top_n_universe': args.top_n_universe,
            'target_horizon_signals': args.target_horizon_signals,
            'target_min_resolved': args.target_min_resolved,
            'target_sl_rate_threshold': args.target_sl_rate_threshold,
            'trade_replay_source': args.trade_replay_source,
            'target_profile': args.target_profile,
            'feature_profile': args.feature_profile,
            'symbols_file': args.symbols_file,
            'fixed_universe_file': args.fixed_universe_file,
            'strategy_state_mode': STRATEGY_STATE_MODE,
            'signals_source_type': source_meta['signals_source_type'],
            'base_model_oos_validated': source_meta['base_model_oos_validated'],
        }
        with open(run_dir / "run_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)

        target_col = args.target_col
        summary = {
            'n_signals': len(signals_df),
            'n_samples': len(dataset),
            'n_features': len(dataset.columns),
            'target_col': target_col,
            'tp_count': int((dataset['trade_outcome'] == 'TP').sum()) if 'trade_outcome' in dataset.columns else None,
            'sl_count': int((dataset['trade_outcome'] == 'SL').sum()) if 'trade_outcome' in dataset.columns else None,
        }
        if target_col in dataset.columns:
            valid = dataset[target_col].dropna()
            summary['target_rate'] = float(valid.mean()) if len(valid) > 0 else None
            summary['target_valid_count'] = len(valid)
            summary['target_nan_count'] = int(dataset[target_col].isna().sum())

        with open(run_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        na_report = dataset.isnull().sum().sort_values(ascending=False)
        na_report = na_report[na_report > 0]
        if len(na_report) > 0:
            na_report.to_csv(run_dir / "feature_na_report.csv")

        if args.save_debug_sample and len(dataset) > 10:
            dataset.head(50).to_parquet(run_dir / "debug_sample.parquet", index=False)
            log("INFO", "REGIME-DS", "saved debug sample to debug_sample.parquet")

        signals_hash = ''
        try:
            with open(args.signals_path, 'rb') as sf:
                signals_hash = hashlib.sha256(sf.read()).hexdigest()[:16]
        except Exception:
            pass

        universe_hash = ''
        universe_path = run_dir / "liquid_universe.json"
        if universe_path.exists():
            try:
                with open(universe_path, 'rb') as uf:
                    universe_hash = hashlib.sha256(uf.read()).hexdigest()[:16]
            except Exception:
                pass

        available_target_columns = sorted([
            c for c in dataset.columns if c.startswith('target_')
        ])

        manifest = {
            'signals_path': args.signals_path,
            'signals_hash': signals_hash,
            'liquid_universe_hash': universe_hash,
            'tp_pct': args.tp_pct,
            'sl_pct': args.sl_pct,
            'max_horizon_bars': args.max_horizon_bars,
            'entry_shift_bars': ENTRY_SHIFT_BARS,
            'bar_minutes': BAR_MINUTES,
            'target_horizon_signals': args.target_horizon_signals,
            'target_min_resolved': args.target_min_resolved,
            'target_sl_rate_threshold': args.target_sl_rate_threshold,
            'target_col': args.target_col,
            'top_n_universe': args.top_n_universe,
            'n_signals': len(signals_df),
            'n_samples': len(dataset),
            'columns': sorted(list(dataset.columns)),
            'available_target_columns': available_target_columns,
            'feature_version': 'v4',
            'feature_profile': args.feature_profile or 'regime_compact_v4',
            'trade_replay_source': args.trade_replay_source,
            'target_profile': args.target_profile,
            'symbols_file': args.symbols_file,
            'fixed_universe_file': args.fixed_universe_file,
            'strategy_state_mode': STRATEGY_STATE_MODE,
            'signals_source_type': source_meta['signals_source_type'],
            'base_model_oos_validated': source_meta['base_model_oos_validated'],
        }
        with open(run_dir / "dataset_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        log("INFO", "REGIME-DS", "saved dataset_manifest.json")


if __name__ == "__main__":
    main()
