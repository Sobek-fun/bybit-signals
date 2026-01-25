import argparse
from datetime import datetime

from pump_end_prod.infra.logging import log


def run_pump_end(args):
    from pump_end_prod.pump_end.pipeline import PumpEndPipeline
    from pump_end_prod.infra.clickhouse import list_all_usdt_tokens

    tokens = [token.strip().upper() for token in args.token.split(",")]

    if len(tokens) == 1 and tokens[0] == "ALL":
        tokens = list_all_usdt_tokens(args.ch_dsn)
        log("INFO", "PUMP_END", f"loaded {len(tokens)} tokens from ClickHouse")

    ws_info = ""
    if args.ws_host and args.ws_port:
        ws_info = f" ws={args.ws_host}:{args.ws_port}"

    log("INFO", "PUMP_END",
        f"start tokens={len(tokens)} workers={args.workers} offset={args.offset_seconds} "
        f"model_dir={args.model_dir} dry_run={args.dry_run}{ws_info}")

    pipeline = PumpEndPipeline(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        model_dir=args.model_dir,
        workers=args.workers,
        offset_seconds=args.offset_seconds,
        dry_run=args.dry_run,
        ws_host=args.ws_host,
        ws_port=args.ws_port
    )

    pipeline.run()


def run_pump_end_export(args):
    from pump_end_prod.pump_end.export_signals import export_signals
    from pump_end_prod.infra.clickhouse import list_all_usdt_tokens

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
    parser = argparse.ArgumentParser(description="Pump End Production Service")
    subparsers = parser.add_subparsers(dest='command', help='Service to run')

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
    pump_end_parser.add_argument(
        "--ws-host",
        type=str,
        default=None,
        help="WebSocket server host (default: disabled)"
    )
    pump_end_parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="WebSocket server port (default: disabled)"
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

    if args.command == 'pump_end':
        run_pump_end(args)
    elif args.command == 'pump_end_export':
        run_pump_end_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
