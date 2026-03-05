import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
from pump_end_threshold.infra.clickhouse import DataLoader, get_liquid_universe
from pump_end_threshold.ml.regime_dataset import (
    build_strategy_state,
    compute_strategy_state_at,
    compute_targets,
    BAR_MINUTES,
    ENTRY_SHIFT_BARS,
)


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


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
    parser.add_argument("--target-col", type=str, default="target_bad_next_5",
                        help="Name of the target column to use for training")
    parser.add_argument("--include-detector-snapshot", action="store_true",
                        help="Include detector snapshot features from PumpFeatureBuilder")
    parser.add_argument("--high-lookback-bars", type=int, default=None,
                        help="Lookback bars for high calculations (default: 8 weeks)")
    parser.add_argument("--breadth-lookback-bars", type=int, default=None,
                        help="Lookback bars for breadth calculations (default: 8 weeks)")
    parser.add_argument("--save-debug-sample", action="store_true",
                        help="Save debug sample of dataset")

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
    t_max = signals_df['open_time'].max()

    log("INFO", "REGIME-DS", "fetching liquid universe")
    liquid_universe = get_liquid_universe(
        args.clickhouse_dsn, t_min - timedelta(days=7), t_max,
        top_n=args.top_n_universe,
    )
    log("INFO", "REGIME-DS", f"liquid universe: {len(liquid_universe)} symbols")

    log("INFO", "REGIME-DS", "simulating trades")
    trades_df = build_strategy_state(
        signals_df, loader,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        max_horizon_bars=args.max_horizon_bars
    )
    log("INFO", "REGIME-DS", f"trades simulated: {len(trades_df)}")

    if 'trade_outcome' in trades_df.columns:
        tp_count = (trades_df['trade_outcome'] == 'TP').sum()
        sl_count = (trades_df['trade_outcome'] == 'SL').sum()
        unknown = (trades_df['trade_outcome'] == 'UNKNOWN').sum()
        timeout = (trades_df['trade_outcome'] == 'TIMEOUT').sum()
        log("INFO", "REGIME-DS", f"outcomes: TP={tp_count} SL={sl_count} UNKNOWN={unknown} TIMEOUT={timeout}")

    log("INFO", "REGIME-DS", "building regime features")

    if args.high_lookback_bars or args.breadth_lookback_bars:
        from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
        if args.high_lookback_bars:
            RegimeFeatureBuilder.HIGH_8W_BARS = args.high_lookback_bars

    builder = RegimeFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        liquid_universe=liquid_universe,
        top_n=args.top_n_universe,
    )

    all_signal_times = signals_df['open_time'].tolist()
    trades_sorted = trades_df.sort_values('open_time').reset_index(drop=True)

    log("INFO", "REGIME-DS", "building regime features in batch mode")
    feature_rows = []
    batch_size = 50

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

    for batch_start in range(0, len(signals_df), batch_size):
        batch_end = min(batch_start + batch_size, len(signals_df))
        batch_signals = signals_df.iloc[batch_start:batch_end]

        batch_features = []
        for i in range(batch_start, batch_end):
            sig = signals_df.iloc[i]
            t = sig['open_time']

            strategy_state = compute_strategy_state_at(trades_sorted, t, all_signal_times)

            sig_for_builder = pd.DataFrame([sig])
            feats = builder.build(sig_for_builder, strategy_state=strategy_state)

            if pump_builder and not feats.empty:
                sig_input = sig_for_builder.copy()
                sig_input = sig_input.rename(columns={'open_time': 'event_open_time'})
                sig_input['pump_la_type'] = 'A'
                sig_input['runup_pct'] = 0
                try:
                    pump_feats = pump_builder.build(sig_input, max_workers=1)
                    if not pump_feats.empty:
                        for col in pump_feats.columns:
                            if col not in feats.columns:
                                feats[col] = pump_feats[col].values[0] if len(pump_feats) > 0 else np.nan
                except:
                    pass
            if feats.empty:
                continue

            targets = compute_targets(
                trades_sorted, i,
                window_sizes=[3, args.target_horizon_signals]
            )
            for k, v in targets.items():
                feats[k] = v

            trade_row = trades_sorted[(trades_sorted['open_time'] == t) & (trades_sorted['symbol'] == sig['symbol'])]
            if not trade_row.empty:
                tr = trade_row.iloc[0]
                feats['trade_outcome'] = tr['trade_outcome']
                feats['tp_hit'] = tr['tp_hit']
                feats['sl_hit'] = tr['sl_hit']
                feats['exit_time'] = tr['exit_time']
                feats['entry_price'] = tr['entry_price']
                feats['exit_price'] = tr['exit_price']
                feats['pnl_pct'] = tr['pnl_pct']
                feats['mfe_pct'] = tr['mfe_pct']
                feats['mae_pct'] = tr['mae_pct']
                feats['trade_duration_bars'] = tr['trade_duration_bars']

            for col in ['p_end_at_fire', 'threshold_gap', 'pending_bars',
                         'drop_from_peak_at_fire', 'signal_offset',
                         'p_end_peak_before_fire', 'threshold_used']:
                if col in sig.index:
                    feats[f'det_{col}'] = sig[col]

            feats['signal_id'] = sig['signal_id']
            batch_features.append(feats)

        feature_rows.extend(batch_features)
        log("INFO", "REGIME-DS", f"processed {batch_end}/{len(signals_df)} signals")

    if not feature_rows:
        log("WARN", "REGIME-DS", "no features built")
        return

    dataset = pd.concat(feature_rows, ignore_index=True)
    log("INFO", "REGIME-DS", f"dataset shape: {dataset.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(out_path), index=False)
    log("INFO", "REGIME-DS", f"saved to {out_path}")

    if 'target_bad_next_5' in dataset.columns:
        bad_rate = dataset['target_bad_next_5'].mean()
        log("INFO", "REGIME-DS", f"target_bad_next_5 rate: {bad_rate:.3f}")

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
        }
        with open(run_dir / "run_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)

        summary = {
            'n_signals': len(signals_df),
            'n_samples': len(dataset),
            'n_features': len(dataset.columns),
            'target_bad_next_5_rate': float(dataset['target_bad_next_5'].mean()) if 'target_bad_next_5' in dataset.columns else None,
            'tp_count': int((dataset['trade_outcome'] == 'TP').sum()) if 'trade_outcome' in dataset.columns else None,
            'sl_count': int((dataset['trade_outcome'] == 'SL').sum()) if 'trade_outcome' in dataset.columns else None,
        }
        with open(run_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        na_report = dataset.isnull().sum().sort_values(ascending=False)
        na_report = na_report[na_report > 0]
        if len(na_report) > 0:
            na_report.to_csv(run_dir / "feature_na_report.csv")

        if args.save_debug_sample and len(dataset) > 10:
            dataset.head(50).to_parquet(run_dir / "debug_sample.parquet", index=False)
            log("INFO", "REGIME-DS", "saved debug sample to debug_sample.parquet")


if __name__ == "__main__":
    main()
