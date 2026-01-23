import argparse
from datetime import datetime

from src.shared.logging import log


def run_pump_start(args):
    from src.config import Config
    from src.prod.pump_start.pipeline import Pipeline
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
        workers=args.workers
    )

    log("INFO", "PUMP_START",
        f"start tokens={len(config.tokens)} workers={config.workers} offset={config.offset_seconds}")

    if len(config.tokens) == 0:
        log("WARN", "PUMP_START", "no tokens specified")

    if config.workers > len(config.tokens):
        log("WARN", "PUMP_START", f"workers({config.workers}) > tokens({len(config.tokens)})")

    pipeline = Pipeline(config)
    pipeline.run()


def run_pump_end(args):
    from src.prod.pump_end.pipeline import PumpEndPipeline
    from src.shared.clickhouse import list_all_usdt_tokens

    tokens = [token.strip().upper() for token in args.token.split(",")]

    if len(tokens) == 1 and tokens[0] == "ALL":
        tokens = list_all_usdt_tokens(args.ch_dsn)
        log("INFO", "PUMP_END", f"loaded {len(tokens)} tokens from ClickHouse")

    log("INFO", "PUMP_END",
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


def run_pump_end_export(args):
    from src.prod.pump_end.export_signals import export_signals
    from src.shared.clickhouse import list_all_usdt_tokens

    tokens = [token.strip().upper() for token in args.token.split(",")]

    if len(tokens) == 1 and tokens[0] == "ALL":
        tokens = list_all_usdt_tokens(args.ch_dsn)
        log("INFO", "PUMP_END_EXPORT", f"loaded {len(tokens)} tokens from ClickHouse")

    dt_from = datetime.strptime(args.dt_from, '%Y-%m-%d %H:%M:%S')
    dt_to = datetime.strptime(args.dt_to, '%Y-%m-%d %H:%M:%S')

    if args.out:
        out_csv = args.out
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = f"pump_end_signals_{timestamp}.csv"

    export_signals(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        model_dir=args.model_dir,
        dt_from=dt_from,
        dt_to=dt_to,
        out_csv=out_csv,
        workers=args.workers
    )


def main():
    parser = argparse.ArgumentParser(description="Production Signal Service")
    subparsers = parser.add_subparsers(dest='command', help='Service to run')

    pump_start_parser = subparsers.add_parser('pump_start', help='Run pump start detection')
    pump_start_parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to monitor (comma-separated, e.g., btc,eth,sol or ALL)"
    )
    pump_start_parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    pump_start_parser.add_argument(
        "--bot-token",
        type=str,
        required=True,
        help="Telegram bot token"
    )
    pump_start_parser.add_argument(
        "--chat-id",
        type=str,
        required=True,
        help="Telegram chat ID"
    )
    pump_start_parser.add_argument(
        "--ws-host",
        type=str,
        default="0.0.0.0",
        help="WebSocket host (default: 0.0.0.0)"
    )
    pump_start_parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)"
    )
    pump_start_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    pump_end_parser = subparsers.add_parser('pump_end', help='Run pump end detection')
    pump_end_parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to monitor (comma-separated, e.g., btc,eth,sol or ALL)"
    )
    pump_end_parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    pump_end_parser.add_argument(
        "--bot-token",
        type=str,
        required=True,
        help="Telegram bot token"
    )
    pump_end_parser.add_argument(
        "--chat-id",
        type=str,
        required=True,
        help="Telegram chat ID"
    )
    pump_end_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory"
    )
    pump_end_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    pump_end_parser.add_argument(
        "--offset-seconds",
        type=int,
        default=3,
        help="Seconds to wait after candle close (default: 3)"
    )
    pump_end_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send Telegram messages, only log"
    )

    pump_end_export_parser = subparsers.add_parser('pump_end_export', help='Export historical pump end signals to CSV')
    pump_end_export_parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token(s) to process (comma-separated, e.g., btc,eth,sol or ALL)"
    )
    pump_end_export_parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    pump_end_export_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory"
    )
    pump_end_export_parser.add_argument(
        "--from",
        type=str,
        required=True,
        dest="dt_from",
        help="Start datetime (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    pump_end_export_parser.add_argument(
        "--to",
        type=str,
        required=True,
        dest="dt_to",
        help="End datetime (YYYY-MM-DD HH:MM:SS), inclusive"
    )
    pump_end_export_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: pump_end_signals_YYYYMMDD_HHMMSS.csv)"
    )
    pump_end_export_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    if args.command == 'pump_start':
        run_pump_start(args)
    elif args.command == 'pump_end':
        run_pump_end(args)
    elif args.command == 'pump_end_export':
        run_pump_end_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
