import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pump_end_threshold.infra.clickhouse import DataLoader
from pump_end_threshold.ml.regime_dataset import build_strategy_state


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def load_symbols_from_file(path: str) -> list[str]:
    lines = Path(path).read_text().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        return symbol
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return symbol


def symbol_to_token(symbol: str) -> str:
    return normalize_symbol(symbol)[:-4]


def load_raw_signals(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=['symbol', 'open_time'])

    raw_signals = pd.read_csv(csv_path)
    if raw_signals.empty:
        return pd.DataFrame(columns=['symbol', 'open_time'])

    if 'timestamp' in raw_signals.columns and 'open_time' not in raw_signals.columns:
        raw_signals = raw_signals.rename(columns={'timestamp': 'open_time'})

    if 'open_time' in raw_signals.columns:
        raw_signals['open_time'] = pd.to_datetime(raw_signals['open_time'])

    return raw_signals


def write_signals_csv(signals_df: pd.DataFrame, output_path: str):
    with open(output_path, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['symbol', 'timestamp'])
        for _, row in signals_df.iterrows():
            timestamp_str = pd.to_datetime(row['open_time']).strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([row['symbol'], timestamp_str])


def resolve_export_tokens(symbols_file: str = None, symbols_csv: str = None, ch_dsn: str = None) -> list[str]:
    if symbols_file:
        symbols = [normalize_symbol(symbol) for symbol in load_symbols_from_file(symbols_file)]
        return [symbol_to_token(symbol) for symbol in symbols]

    if symbols_csv:
        symbols = [normalize_symbol(symbol) for symbol in symbols_csv.split(',') if symbol.strip()]
        return [symbol_to_token(symbol) for symbol in symbols]

    from pump_end_prod.infra.clickhouse import list_all_usdt_tokens
    return list_all_usdt_tokens(ch_dsn)


def run_guard_stage(
        raw_signals_df: pd.DataFrame,
        guard_model_dir: Path,
        ch_dsn: str,
        run_dir: Path = None,
        guard_debug_output: str = None,
        blocked_signals_output: str = None,
        accepted_signals_output: str = None,
) -> pd.DataFrame:
    from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
    from pump_end_threshold.ml.regime_policy import RegimePolicy
    from pump_end_threshold.infra.clickhouse import get_liquid_universe

    guard_model_path = guard_model_dir / "regime_guard_model.cbm"
    guard_model = CatBoostClassifier()
    guard_model.load_model(str(guard_model_path))

    policy_path = guard_model_dir / "best_policy_params.json"
    with open(policy_path, 'r') as f:
        policy_params = json.load(f)

    feature_cols_path = guard_model_dir / "feature_columns.json"
    with open(feature_cols_path, 'r') as f:
        guard_feature_columns = json.load(f)

    signals = raw_signals_df.sort_values('open_time').reset_index(drop=True)
    t_min = signals['open_time'].min()
    t_max = signals['open_time'].max()

    liquid_universe_path = guard_model_dir / "liquid_universe.json"
    if liquid_universe_path.exists():
        log("INFO", "GUARD", f"loading liquid_universe from {liquid_universe_path}")
        with open(liquid_universe_path, 'r') as f:
            liquid_universe = json.load(f)
    else:
        log("WARN", "GUARD", "liquid_universe.json not found, computing from scratch")
        liquid_universe = get_liquid_universe(
            ch_dsn, t_min - timedelta(days=7), t_max, top_n=120
        )

    regime_config_path = guard_model_dir / "regime_builder_config.json"
    if regime_config_path.exists():
        log("INFO", "GUARD", f"loading regime builder config from {regime_config_path}")
        with open(regime_config_path, 'r') as f:
            regime_config = json.load(f)
        top_n = regime_config.get('top_n_universe', 120)
    else:
        log("WARN", "GUARD", "regime_builder_config.json not found, using defaults")
        top_n = 120

    builder = RegimeFeatureBuilder(
        ch_dsn=ch_dsn,
        liquid_universe=liquid_universe,
        top_n=top_n,
    )

    loader = DataLoader(ch_dsn)
    tp_pct = 4.5
    sl_pct = 10.0
    max_horizon_bars = 200
    manifest_path = guard_model_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        tp_pct = manifest.get('tp_pct', tp_pct)
        sl_pct = manifest.get('sl_pct', sl_pct)
        max_horizon_bars = manifest.get('max_horizon_bars', max_horizon_bars)

    log("INFO", "GUARD", "simulating trades for strategy state features")
    trades_df = build_strategy_state(
        signals, loader,
        tp_pct=tp_pct, sl_pct=sl_pct,
        max_horizon_bars=max_horizon_bars,
    )
    log("INFO", "GUARD", f"trades simulated: {len(trades_df)}")

    log("INFO", "GUARD", "building regime features in batch mode")
    guard_features = builder.build_batch(signals, batch_size=100, trades_df=trades_df)

    if guard_features.empty:
        log("WARN", "GUARD", "no features built")
        return raw_signals_df

    available = [c for c in guard_feature_columns if c in guard_features.columns]
    if len(available) < len(guard_feature_columns):
        missing = set(guard_feature_columns) - set(available)
        log("WARN", "GUARD", f"missing {len(missing)} feature columns, filling with NaN: {list(missing)[:10]}")
        for col in missing:
            guard_features[col] = np.nan

    X_guard = guard_features[guard_feature_columns]
    p_bad = guard_model.predict_proba(X_guard)[:, 1]

    signals['p_bad'] = p_bad

    if guard_debug_output:
        signals.to_parquet(guard_debug_output, index=False)
    elif run_dir:
        guard_scored_path = run_dir / "guard_scored_signals.parquet"
        signals.to_parquet(guard_scored_path, index=False)

    if 'pause_on_quantile' in policy_params:
        resolved = {
            'pause_on_threshold': float(np.quantile(p_bad, policy_params['pause_on_quantile'])),
            'resume_threshold': float(np.quantile(p_bad, policy_params['resume_quantile'])),
            'resume_confirm_signals': policy_params['resume_confirm_signals'],
        }
        log("INFO", "GUARD",
            f"resolved quantile policy: pause={resolved['pause_on_threshold']:.4f} "
            f"resume={resolved['resume_threshold']:.4f}")
        policy = RegimePolicy(**resolved)
    else:
        policy = RegimePolicy(**policy_params)
    result = policy.apply(signals, p_bad_col='p_bad')

    accepted = result[~result['blocked_by_policy']].copy()
    blocked = result[result['blocked_by_policy']].copy()

    if accepted_signals_output:
        accepted.to_parquet(accepted_signals_output, index=False)
    elif run_dir:
        accepted.to_parquet(run_dir / "accepted_signals.parquet", index=False)

    if blocked_signals_output:
        blocked.to_parquet(blocked_signals_output, index=False)
    elif run_dir:
        blocked.to_parquet(run_dir / "blocked_signals.parquet", index=False)

    n_blocked = result['blocked_by_policy'].sum()
    log("INFO", "GUARD", f"guard applied: {len(result)} raw -> {len(accepted)} accepted ({n_blocked} blocked)")

    return accepted


def main():
    parser = argparse.ArgumentParser(description="Export pump end signals from trained model")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    parser.add_argument(
        "--clickhouse-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN (e.g., http://user:pass@host:port/database)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to detector model artifacts directory"
    )
    parser.add_argument(
        "--guard-model-dir",
        type=str,
        default=None,
        help="Path to regime guard model directory (optional, enables guard stage)"
    )
    parser.add_argument(
        "--skip-guard",
        action="store_true",
        help="Skip guard stage even if guard model exists in run-dir"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for all artifacts (if not provided, uses individual output paths)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: RUN_DIR/final_signals.csv or pump_end_signals.csv)"
    )
    parser.add_argument(
        "--raw-signals-output",
        type=str,
        default=None,
        help="Optional: save raw detector signals to this parquet path (default: RUN_DIR/raw_detector_signals.parquet if --run-dir set)"
    )
    parser.add_argument(
        "--guard-debug-output",
        type=str,
        default=None,
        help="Optional: save guard scored signals to this parquet path"
    )
    parser.add_argument(
        "--blocked-signals-output",
        type=str,
        default=None,
        help="Optional: save blocked signals to this parquet path"
    )
    parser.add_argument(
        "--accepted-signals-output",
        type=str,
        default=None,
        help="Optional: save accepted signals to this parquet path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Path to file with allowed symbols (one per line)"
    )
    parser.add_argument(
        "--symbols-csv",
        type=str,
        default=None,
        help="Comma-separated list of allowed symbols (e.g., 'BTCUSDT,ETHUSDT,...')"
    )

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M:%S')
    model_dir = Path(args.model_dir)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = run_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        if not args.output:
            args.output = str(run_dir / "final_signals.csv")
        if not args.raw_signals_output:
            args.raw_signals_output = str(run_dir / "raw_detector_signals.parquet")
    else:
        run_dir = None
        temp_dir = Path(".")
        if not args.output:
            args.output = "pump_end_signals.csv"

    tokens = resolve_export_tokens(
        symbols_file=args.symbols_file,
        symbols_csv=args.symbols_csv,
        ch_dsn=args.clickhouse_dsn,
    )

    if not tokens:
        log("WARN", "EXPORT", "no tokens resolved for export")
        write_signals_csv(pd.DataFrame(columns=['symbol', 'open_time']), args.output)
        if args.raw_signals_output:
            pd.DataFrame(columns=['symbol', 'open_time']).to_parquet(args.raw_signals_output, index=False)
        return

    stage_a_csv = temp_dir / "prod_raw_signals.csv"

    log("INFO", "EXPORT", f"stage A: replaying prod exporter for {len(tokens)} tokens")
    from pump_end_prod.pump_end.export_signals import export_signals
    export_signals(
        tokens=tokens,
        ch_dsn=args.clickhouse_dsn,
        model_dir=str(model_dir),
        dt_from=start_dt,
        dt_to=end_dt,
        out_csv=str(stage_a_csv),
        workers=args.workers,
    )

    raw_signals = load_raw_signals(stage_a_csv)
    log("INFO", "EXPORT", f"stage A done: {len(raw_signals)} raw detector signals")

    if args.raw_signals_output:
        raw_signals.to_parquet(args.raw_signals_output, index=False)
        log("INFO", "EXPORT", f"raw signals saved to {args.raw_signals_output}")

    if args.skip_guard:
        log("INFO", "EXPORT", "stage B: skipping guard stage as requested")
        final_signals = raw_signals
    elif args.guard_model_dir:
        guard_dir = Path(args.guard_model_dir)
        log("INFO", "EXPORT", f"stage B: applying regime guard from {guard_dir}")
        final_signals = run_guard_stage(
            raw_signals, guard_dir, args.clickhouse_dsn, run_dir=run_dir,
            guard_debug_output=args.guard_debug_output,
            blocked_signals_output=args.blocked_signals_output,
            accepted_signals_output=args.accepted_signals_output
        )
    elif run_dir and (run_dir / "regime_guard_model.cbm").exists():
        log("INFO", "EXPORT", f"stage B: applying regime guard from {run_dir}")
        final_signals = run_guard_stage(
            raw_signals, run_dir, args.clickhouse_dsn, run_dir=run_dir,
            guard_debug_output=args.guard_debug_output,
            blocked_signals_output=args.blocked_signals_output,
            accepted_signals_output=args.accepted_signals_output
        )
    else:
        log("INFO", "EXPORT", "stage B: no guard model, passing all raw signals through")
        final_signals = raw_signals

    write_signals_csv(final_signals, args.output)

    log("INFO", "EXPORT",
        f"done tokens={len(tokens)} "
        f"raw_signals={len(raw_signals)} final_signals={len(final_signals)} output={args.output}")


if __name__ == "__main__":
    main()
