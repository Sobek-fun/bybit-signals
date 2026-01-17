import argparse
from datetime import datetime
from urllib.parse import urlparse

import clickhouse_connect

from src.monitoring.pump_end_pipeline import PumpEndPipeline


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def main():
    parser = argparse.ArgumentParser(description="Pump End Model Inference Service")
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
        help="Path to model directory (contains catboost_model.cbm, best_threshold.json)"
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

    args = parser.parse_args()

    tokens = [token.strip().upper() for token in args.token.split(",")]

    if len(tokens) == 1 and tokens[0] == "ALL":
        parsed = urlparse(args.ch_dsn)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8123
        username = parsed.username or "default"
        password = parsed.password or ""
        database = parsed.path.lstrip("/") if parsed.path else "default"
        secure = parsed.scheme == "https"

        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            secure=secure
        )

        query = "SELECT DISTINCT symbol FROM bybit.transactions WHERE endsWith(symbol, 'USDT') ORDER BY symbol"
        result = client.query(query)
        tokens = [row[0][:-4] for row in result.result_rows]
        log("INFO", "MAIN", f"loaded {len(tokens)} tokens from ClickHouse")

    log("INFO", "MAIN",
        f"start tokens={len(tokens)} workers={args.workers} offset={args.offset_seconds} "
        f"model_dir={args.model_dir} dry_run={args.dry_run}")

    pipeline = PumpEndPipeline(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        model_dir=args.model_dir,
        workers=args.workers,
        offset_seconds=args.offset_seconds,
        dry_run=args.dry_run
    )

    pipeline.run()


if __name__ == "__main__":
    main()
