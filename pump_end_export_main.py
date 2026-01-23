import argparse
from datetime import datetime

from src.shared.logging import log


def main():
    parser = argparse.ArgumentParser(description="Export historical Pump End ML signals to CSV")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to process (comma-separated, e.g., btc,eth,sol or ALL)"
    )
    parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory (contains catboost_model.cbm, best_threshold.json)"
    )
    parser.add_argument(
        "--from",
        type=str,
        required=True,
        dest="dt_from",
        help="Start datetime (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    parser.add_argument(
        "--to",
        type=str,
        required=True,
        dest="dt_to",
        help="End datetime (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: pump_end_signals_YYYYMMDD_HHMMSS.csv)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    from src.prod.cli import run_pump_end_export
    run_pump_end_export(args)


if __name__ == "__main__":
    main()
