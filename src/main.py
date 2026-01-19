import argparse
import sys
from datetime import datetime

from src.shared.logging import log


def main():
    parser = argparse.ArgumentParser(description="Pump Monitoring Service")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prod", "test", "debug"],
        default="prod",
        help="Run mode: prod (continuous monitoring), test (historical backtest), or debug (indicator snapshot)"
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
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp for debug mode (format: YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--timestamp-kind",
        type=str,
        choices=["close", "bucket"],
        default="close",
        help="Timestamp interpretation: 'close' (close_time) or 'bucket' (bucket_start)"
    )

    args = parser.parse_args()

    if args.mode == "prod":
        if not args.bot_token or not args.chat_id:
            parser.error("--bot-token and --chat-id are required for prod mode")

        from src.prod.cli import run_pump_start
        run_pump_start(args)

    elif args.mode == "test":
        from src.config import Config
        from src.dev.tools.test_runner import TestRunner
        from src.shared.clickhouse import list_all_usdt_tokens

        tokens = [token.strip().upper() for token in args.token.split(",")]

        if len(tokens) == 1 and tokens[0] == "ALL":
            tokens = list_all_usdt_tokens(args.ch_dsn)

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

        log("INFO", "MAIN",
            f"mode=test tokens={len(config.tokens)} days={config.test_days} lookback={config.lookback_candles}")

        runner = TestRunner(config)
        runner.run_test()

    elif args.mode == "debug":
        if not args.timestamp:
            parser.error("--timestamp is required for debug mode")

        from src.dev.tools.indicator_snapshot import get_indicator_snapshot

        tokens = [token.strip() for token in args.token.split(",")]
        if len(tokens) != 1:
            parser.error("debug mode requires exactly one token")

        symbol = tokens[0].upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"

        get_indicator_snapshot(
            ch_dsn=args.ch_dsn,
            symbol=symbol,
            timestamp_str=args.timestamp,
            timestamp_kind=args.timestamp_kind,
            lookback_candles=150
        )


if __name__ == "__main__":
    main()
