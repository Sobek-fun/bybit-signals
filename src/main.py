import argparse
from datetime import datetime
from urllib.parse import urlparse

import clickhouse_connect

from src.config import Config
from src.monitoring.pipeline import Pipeline


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def main():
    parser = argparse.ArgumentParser(description="Pump Monitoring Service")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prod", "test"],
        default="prod",
        help="Run mode: prod (continuous monitoring) or test (historical backtest)"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to monitor (comma-separated, e.g., btc,eth,sol)"
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
        help="Telegram bot token (required for prod mode)"
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        help="Telegram chat ID (required for prod mode)"
    )
    parser.add_argument(
        "--ws-host",
        type=str,
        default="0.0.0.0",
        help="WebSocket host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8, prod mode only)"
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Number of days to backtest (default: 30, test mode only)"
    )

    args = parser.parse_args()

    if args.mode == "prod":
        if not args.bot_token or not args.chat_id:
            parser.error("--bot-token and --chat-id are required for prod mode")

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

    config = Config(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        bot_token=args.bot_token or "",
        chat_id=args.chat_id or "",
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        workers=args.workers,
        test_days=args.test_days
    )

    if args.mode == "test":
        from src.testing.test_runner import TestRunner

        log("INFO", "MAIN",
            f"mode=test tokens={len(config.tokens)} days={config.test_days} lookback={config.lookback_candles}")

        runner = TestRunner(config)
        runner.run_test()
    else:
        parsed = urlparse(config.ch_dsn)
        ch_host = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 8123}"
        ch_db = parsed.path.lstrip("/") if parsed.path else "default"

        log("INFO", "MAIN",
            f"start tokens={len(config.tokens)} workers={config.workers} offset={config.offset_seconds} lookback={config.lookback_candles}")
        log("INFO", "MAIN", f"clickhouse={ch_host}/{ch_db} chat_id={config.chat_id}")

        if len(config.tokens) == 0:
            log("WARN", "MAIN", "no tokens specified")

        if config.workers > len(config.tokens):
            log("WARN", "MAIN", f"workers({config.workers}) > tokens({len(config.tokens)})")

        pipeline = Pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
