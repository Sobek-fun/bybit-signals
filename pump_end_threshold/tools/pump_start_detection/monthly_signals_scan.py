import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from pump_end.tools.pump_start_detection.indicators import IndicatorCalculator
from pump_end.tools.pump_start_detection.detector import PumpDetector


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def get_last_closed_time(client, offset_seconds: int):
    now = datetime.now()
    minutes = (now.minute // 15) * 15
    current_bucket_start = now.replace(minute=minutes, second=0, microsecond=0)

    if now.minute % 15 == 0 and now.second < offset_seconds:
        current_bucket_start = current_bucket_start - timedelta(minutes=15)

    query = """
    SELECT open_time
    FROM bybit.candles
    WHERE interval = 1
      AND open_time < %(current_bucket_start)s
    ORDER BY open_time DESC
    LIMIT 1
    """

    result = client.query(query, parameters={
        "current_bucket_start": current_bucket_start
    })

    if not result.result_rows:
        return None

    bucket = result.result_rows[0][0]
    return bucket + timedelta(minutes=1)


def get_available_symbols(client, query_start_bucket: datetime, end_close_time: datetime):
    query = """
    SELECT DISTINCT symbol
    FROM bybit.candles
    WHERE interval = 1
      AND open_time >= %(start)s
      AND open_time < %(end)s
      AND symbol LIKE '%%USDT'
    ORDER BY symbol
    """

    result = client.query(query, parameters={
        "start": query_start_bucket,
        "end": end_close_time
    })

    return [row[0] for row in result.result_rows]


def process_symbol_chunk(worker_id: int, symbols_chunk: list, ch_dsn: str, query_start_bucket: datetime,
                         end_bucket: datetime, start_bucket: datetime, lookback_candles: int, offset_seconds: int):
    parsed = urlparse(ch_dsn)
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

    calculator = IndicatorCalculator()
    detector = PumpDetector()

    csv_filename = f"signals_part_{worker_id}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'symbol'])

    total_signals = 0
    symbols_processed = 0
    symbols_skipped = 0
    errors = 0

    end_close_time = end_bucket + timedelta(minutes=15)

    for symbol in symbols_chunk:
        try:
            query = """
            SELECT
                open_time AS bucket,
                open,
                high,
                low,
                close,
                volume,
                0 AS buy_volume,
                0 AS sell_volume,
                0 AS net_volume,
                0 AS trades_count
            FROM bybit.candles
            WHERE symbol = %(symbol)s
              AND interval = 1
              AND open_time >= %(start)s
              AND open_time < %(end)s
            ORDER BY open_time
            """

            result = client.query(query, parameters={
                "symbol": symbol,
                "start": query_start_bucket,
                "end": end_close_time
            })

            if not result.result_rows:
                symbols_skipped += 1
                continue

            df = pd.DataFrame(
                result.result_rows,
                columns=["bucket", "open", "high", "low", "close", "volume",
                         "buy_volume", "sell_volume", "net_volume", "trades_count"]
            )

            df["bucket"] = pd.to_datetime(df["bucket"])
            df.set_index("bucket", inplace=True)

            df['bucket15'] = df.index.floor('15min')

            df_15m = df.groupby('bucket15').agg({
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum',
                'buy_volume': 'sum',
                'sell_volume': 'sum',
                'net_volume': 'sum',
                'trades_count': 'sum'
            })

            if len(df_15m) < lookback_candles:
                symbols_skipped += 1
                continue

            df_15m = calculator.calculate(df_15m)
            df_15m = detector.detect(df_15m)

            signals = df_15m[(df_15m.index >= start_bucket) & (df_15m['pump_signal'] == 'strong_pump')]

            for idx, row in signals.iterrows():
                close_time = idx + timedelta(minutes=15)
                timestamp_str = close_time.strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([timestamp_str, symbol])
                total_signals += 1

            symbols_processed += 1

        except Exception as e:
            errors += 1
            if errors == 1:
                log("ERROR", f"WORKER{worker_id}", f"symbol={symbol} error={type(e).__name__}: {str(e)}")

    csv_file.close()

    return worker_id, symbols_processed, symbols_skipped, total_signals, errors, csv_filename


def main():
    parser = argparse.ArgumentParser(description="Monthly Signals Scan")
    parser.add_argument(
        "--ch-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to scan (default: 30)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=150,
        help="Lookback candles for warmup (default: 150)"
    )
    parser.add_argument(
        "--offset-seconds",
        type=int,
        default=10,
        help="Offset seconds for closed candle detection (default: 10)"
    )

    args = parser.parse_args()

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

    start_time = datetime.now()

    end_close_time = get_last_closed_time(client, args.offset_seconds)
    if end_close_time is None:
        log("ERROR", "SCAN", "no closed candles found")
        return

    start_close_time = end_close_time - timedelta(days=args.days)
    end_bucket = end_close_time - timedelta(minutes=15)
    start_bucket = start_close_time - timedelta(minutes=15)
    query_start_bucket = start_bucket - timedelta(minutes=args.lookback * 15)

    symbols = get_available_symbols(client, query_start_bucket, end_close_time)

    if not symbols:
        log("ERROR", "SCAN", "no symbols found")
        return

    num_workers = min(4, len(symbols))
    chunk_size = len(symbols) // num_workers
    remainder = len(symbols) % num_workers

    chunks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(symbols[start_idx:end_idx])
        start_idx = end_idx

    log("INFO", "SCAN",
        f"start days={args.days} end_close_time={end_close_time.strftime('%Y-%m-%d %H:%M:%S')} symbols={len(symbols)} workers={num_workers}")

    part_files = []
    total_processed = 0
    total_skipped = 0
    total_signals = 0
    total_errors = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_symbol_chunk, i + 1, chunks[i], args.ch_dsn, query_start_bucket,
                            end_bucket, start_bucket, args.lookback, args.offset_seconds): i + 1
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
            log("INFO", "SCAN",
                f"worker={worker_id} done symbols_ok={processed} skipped={skipped} signals={signals} errors={errors} time={worker_time:.1f}s")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_csv = f"signals_monthly_{timestamp}.csv"

    with open(final_csv, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['timestamp', 'symbol'])

        for part_csv in part_files:
            if os.path.exists(part_csv):
                with open(part_csv, 'r') as infile:
                    reader = csv.reader(infile)
                    next(reader)
                    for row in reader:
                        csv_writer.writerow(row)
                os.remove(part_csv)

    total_time = (datetime.now() - start_time).total_seconds()

    log("INFO", "SCAN",
        f"done total_symbols={len(symbols)} processed={total_processed} total_signals={total_signals} csv={final_csv} total_time={total_time:.1f}s")


if __name__ == "__main__":
    main()
