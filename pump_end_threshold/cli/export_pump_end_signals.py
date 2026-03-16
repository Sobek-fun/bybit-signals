import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pump_end_threshold.ml.regime_inference import apply_guard_to_raw_signals


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
        final_signals = apply_guard_to_raw_signals(
            raw_signals, guard_dir, args.clickhouse_dsn, run_dir=run_dir,
            guard_debug_output=args.guard_debug_output,
            blocked_signals_output=args.blocked_signals_output,
            accepted_signals_output=args.accepted_signals_output
        )
    elif run_dir and (run_dir / "regime_guard_model.cbm").exists():
        log("INFO", "EXPORT", f"stage B: applying regime guard from {run_dir}")
        final_signals = apply_guard_to_raw_signals(
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
