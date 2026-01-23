import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pump_end_prod.infra.clickhouse import DataLoader
from pump_end_prod.infra.logging import log
from pump_end_prod.pump_end.feature_builder import PumpFeatureBuilder
from pump_end_prod.pump_end.model import PumpEndModel
from pump_end_prod.pump_end.worker import PumpEndSignalState

MIN_CANDLES = 873
WARMUP_BARS = 25


def _process_symbol_chunk(
        worker_id: int,
        symbols_chunk: list,
        ch_dsn: str,
        model_dir: str,
        dt_from: datetime,
        dt_to: datetime
) -> tuple:
    loader = DataLoader(ch_dsn)
    model = PumpEndModel(model_dir)
    feature_builder = PumpFeatureBuilder(
        ch_dsn=None,
        window_bars=model.window_bars,
        warmup_bars=model.warmup_bars,
        feature_set=model.feature_set
    )

    csv_filename = f"signals_part_{worker_id}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['symbol', 'timestamp'])

    total_signals = 0
    symbols_processed = 0
    symbols_skipped = 0
    errors = 0

    warmup_start = dt_from - timedelta(minutes=WARMUP_BARS * 15)

    for symbol in symbols_chunk:
        try:
            result = _process_single_symbol(
                symbol=symbol,
                loader=loader,
                model=model,
                feature_builder=feature_builder,
                warmup_start=warmup_start,
                dt_from=dt_from,
                dt_to=dt_to
            )

            if result is None:
                symbols_skipped += 1
                continue

            for sig_time in result:
                timestamp_str = sig_time.strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([symbol, timestamp_str])
                total_signals += 1

            symbols_processed += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                log("WARN", f"EXPORT_WORKER{worker_id}", f"symbol={symbol} error={type(e).__name__}: {str(e)}")

    csv_file.close()

    return worker_id, symbols_processed, symbols_skipped, total_signals, errors, csv_filename


def _process_single_symbol(
        symbol: str,
        loader: DataLoader,
        model: PumpEndModel,
        feature_builder: PumpFeatureBuilder,
        warmup_start: datetime,
        dt_from: datetime,
        dt_to: datetime
) -> list:
    query_start_bucket = warmup_start - timedelta(minutes=15) - timedelta(minutes=(MIN_CANDLES - 1) * 15)
    end_bucket = dt_to - timedelta(minutes=15)

    df = loader.load_candles_range(symbol, query_start_bucket, end_bucket + timedelta(minutes=15))

    if df.empty:
        return None

    all_decision_times = pd.date_range(
        start=warmup_start,
        end=dt_to,
        freq='15min'
    ).tolist()

    valid_decision_times = []
    for dt in all_decision_times:
        expected_bucket_start = dt - timedelta(minutes=15)
        if expected_bucket_start not in df.index:
            continue

        idx = df.index.get_loc(expected_bucket_start)
        if idx < MIN_CANDLES - 1:
            continue

        start_idx = idx - (MIN_CANDLES - 1)
        expected_range = pd.date_range(
            start=df.index[start_idx],
            end=expected_bucket_start,
            freq='15min'
        )
        actual_range = df.index[start_idx:idx + 1]

        if len(actual_range) == MIN_CANDLES and len(expected_range) == MIN_CANDLES:
            if (actual_range == expected_range).all():
                valid_decision_times.append(dt)

    if not valid_decision_times:
        return None

    feature_rows = feature_builder.build_many_for_inference(df, symbol, valid_decision_times)

    if not feature_rows:
        return None

    feature_values_list = []
    for row in feature_rows:
        fv = []
        for name in model.feature_names:
            val = row.get(name)
            fv.append(np.nan if val is None else val)
        feature_values_list.append(fv)

    probas = model.model.predict_proba(feature_values_list)[:, 1]

    signal_state = PumpEndSignalState()
    triggered_signals = []

    for i, dt in enumerate(valid_decision_times):
        p_end = probas[i]

        triggered = signal_state.update_and_check(
            symbol,
            p_end,
            model.threshold,
            model.min_pending_bars,
            model.drop_delta,
            dt
        )

        if triggered and dt >= dt_from and dt <= dt_to:
            triggered_signals.append(dt)

    return triggered_signals


def export_signals(
        tokens: list,
        ch_dsn: str,
        model_dir: str,
        dt_from: datetime,
        dt_to: datetime,
        out_csv: str,
        workers: int
):
    log("INFO", "EXPORT", f"starting export from={dt_from} to={dt_to} tokens={len(tokens)} workers={workers}")

    dt_from = _align_to_15min(dt_from)
    dt_to = _align_to_15min(dt_to)

    log("INFO", "EXPORT", f"aligned range: from={dt_from} to={dt_to}")

    symbols = [f"{token}USDT" for token in tokens]

    num_workers = min(workers, len(symbols))
    chunk_size = len(symbols) // num_workers
    remainder = len(symbols) % num_workers

    chunks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(symbols[start_idx:end_idx])
        start_idx = end_idx

    part_files = []
    total_processed = 0
    total_skipped = 0
    total_signals = 0
    total_errors = 0

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_symbol_chunk,
                i + 1,
                chunks[i],
                ch_dsn,
                model_dir,
                dt_from,
                dt_to
            ): i + 1
            for i in range(num_workers)
        }

        for future in as_completed(futures):
            worker_id, processed, skipped, signals, errors, csv_file = future.result()
            part_files.append(csv_file)
            total_processed += processed
            total_skipped += skipped
            total_signals += signals
            total_errors += errors

            worker_time = (datetime.now() - start_time).total_seconds()
            log("INFO", "EXPORT",
                f"worker={worker_id} done processed={processed} skipped={skipped} signals={signals} errors={errors} time={worker_time:.1f}s")

    with open(out_csv, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['symbol', 'timestamp'])

        for part_csv in part_files:
            if os.path.exists(part_csv):
                with open(part_csv, 'r') as infile:
                    reader = csv.reader(infile)
                    next(reader)
                    for row in reader:
                        csv_writer.writerow(row)
                os.remove(part_csv)

    total_time = (datetime.now() - start_time).total_seconds()

    log("INFO", "EXPORT",
        f"done total_symbols={len(symbols)} processed={total_processed} skipped={total_skipped} "
        f"signals={total_signals} errors={total_errors} csv={out_csv} time={total_time:.1f}s")


def _align_to_15min(dt: datetime) -> datetime:
    minutes = (dt.minute // 15) * 15
    return dt.replace(minute=minutes, second=0, microsecond=0)
