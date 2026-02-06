from pump_end_clustering_prod.infra.logging import log


def run_pump_end_clustering(args):
    from pump_end_clustering_prod.pump_end.pipeline import PumpEndClusteringPipeline
    from pump_end_clustering_prod.infra.clickhouse import list_all_usdt_tokens

    tokens = [token.strip().upper() for token in args.token.split(",")]

    if len(tokens) == 1 and tokens[0] == "ALL":
        tokens = list_all_usdt_tokens(args.ch_dsn)
        log("INFO", "PUMP_END_CL", f"loaded {len(tokens)} tokens from ClickHouse")

    log("INFO", "PUMP_END_CL",
        f"start tokens={len(tokens)} workers={args.workers} offset={args.offset_seconds} "
        f"model_dir={args.model_dir} dry_run={args.dry_run} restore_lookback={args.restore_lookback_bars}")

    pipeline = PumpEndClusteringPipeline(
        tokens=tokens,
        ch_dsn=args.ch_dsn,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        model_dir=args.model_dir,
        workers=args.workers,
        offset_seconds=args.offset_seconds,
        dry_run=args.dry_run,
        restore_lookback_bars=args.restore_lookback_bars
    )

    pipeline.run()
