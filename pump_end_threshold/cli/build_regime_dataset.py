import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

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
    parser.add_argument("--signals-csv", type=str, required=True,
                        help="Path to historical detector signals CSV (columns: symbol, timestamp or open_time)")
    parser.add_argument("--clickhouse-dsn", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--top-n-universe", type=int, default=120)
    parser.add_argument("--tp-pct", type=float, default=4.5)
    parser.add_argument("--sl-pct", type=float, default=10.0)

    args = parser.parse_args()

    log("INFO", "REGIME-DS", f"loading signals from {args.signals_csv}")
    signals_df = pd.read_csv(args.signals_csv)

    if 'timestamp' in signals_df.columns and 'open_time' not in signals_df.columns:
        signals_df = signals_df.rename(columns={'timestamp': 'open_time'})
    signals_df['open_time'] = pd.to_datetime(signals_df['open_time'])
    signals_df = signals_df.sort_values('open_time').reset_index(drop=True)

    if 'event_id' not in signals_df.columns:
        signals_df['event_id'] = [
            f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
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
    trades_df = build_strategy_state(signals_df, loader, tp_pct=args.tp_pct, sl_pct=args.sl_pct)
    log("INFO", "REGIME-DS", f"trades simulated: {len(trades_df)}")

    if 'trade_outcome' in trades_df.columns:
        tp_count = (trades_df['trade_outcome'] == 'TP').sum()
        sl_count = (trades_df['trade_outcome'] == 'SL').sum()
        unknown = (trades_df['trade_outcome'] == 'UNKNOWN').sum()
        timeout = (trades_df['trade_outcome'] == 'TIMEOUT').sum()
        log("INFO", "REGIME-DS", f"outcomes: TP={tp_count} SL={sl_count} UNKNOWN={unknown} TIMEOUT={timeout}")

    log("INFO", "REGIME-DS", "building regime features")
    builder = RegimeFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        liquid_universe=liquid_universe,
        top_n=args.top_n_universe,
    )

    all_signal_times = signals_df['open_time'].tolist()
    trades_sorted = trades_df.sort_values('open_time').reset_index(drop=True)

    feature_rows = []
    for i in range(len(signals_df)):
        sig = signals_df.iloc[i]
        t = sig['open_time']

        strategy_state = compute_strategy_state_at(trades_sorted, t, all_signal_times)

        sig_for_builder = pd.DataFrame([sig])
        feats = builder.build(sig_for_builder, strategy_state=strategy_state)
        if feats.empty:
            continue

        targets = compute_targets(trades_sorted, i)
        for k, v in targets.items():
            feats[k] = v

        trade_row = trades_sorted[trades_sorted['open_time'] == t]
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

        feature_rows.append(feats)

        if (i + 1) % 50 == 0:
            log("INFO", "REGIME-DS", f"processed {i + 1}/{len(signals_df)} signals")

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


if __name__ == "__main__":
    main()
