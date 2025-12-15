import argparse
from src.config import Config
from src.monitoring.pipeline import Pipeline


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

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
