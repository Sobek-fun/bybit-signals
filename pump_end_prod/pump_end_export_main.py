import argparse

from pump_end_prod.infra.logging import log


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
    parser.add_argument(
        "--regime-on",
        action="store_true",
        default=False,
        help="Enable regime guard stage"
    )
    parser.add_argument(
        "--regime-model-dir",
        type=str,
        default=None,
        help="Path to regime guard artifacts directory"
    )

    args = parser.parse_args()

    if args.regime_on and not args.regime_model_dir:
        raise ValueError("--regime-model-dir is required when --regime-on is set")

    from pump_end_prod.cli import run_pump_end_export
    run_pump_end_export(args)


if __name__ == "__main__":
    main()
