import argparse


def main():
    parser = argparse.ArgumentParser(description="Pump End Clustering Model Inference Service")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to monitor (comma-separated, e.g., btc,eth,sol or ALL)"
    )
    parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    parser.add_argument(
        "--bot-token",
        type=str,
        required=True,
        help="Telegram bot token"
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        required=True,
        help="Telegram chat ID"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory (contains catboost_model.cbm, best_threshold.json, clusters/)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--offset-seconds",
        type=int,
        default=3,
        help="Seconds to wait after candle close (default: 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send Telegram messages, only log"
    )
    parser.add_argument(
        "--restore-lookback-bars",
        type=int,
        default=0,
        help="Number of 15m bars to look back for state restoration on startup (default: 0 = disabled)"
    )

    args = parser.parse_args()

    from pump_end_clustering_prod.cli import run_pump_end_clustering
    run_pump_end_clustering(args)


if __name__ == "__main__":
    main()
