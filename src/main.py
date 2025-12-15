import argparse
from datetime import datetime
from urllib.parse import urlparse

from src.config import Config
from src.monitoring.pipeline import Pipeline


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def main():
    parser = argparse.ArgumentParser(description="Pump Monitoring Service")
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
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    tokens = [token.strip().upper() for token in args.token.split(",")]

    config = Config(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        workers=args.workers
    )

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
