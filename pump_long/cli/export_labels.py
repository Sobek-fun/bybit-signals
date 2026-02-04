import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from pump_long.labeler import PumpStartLabelerLookahead
from pump_long.infra.logging import log

_worker_client_cache = {}


def _get_cached_client(ch_dsn: str):
    if ch_dsn not in _worker_client_cache:
        parsed = urlparse(ch_dsn)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8123
        username = parsed.username or "default"
        password = parsed.password or ""
        database = parsed.path.lstrip("/") if parsed.path else "default"
        secure = parsed.scheme == "https"

        _worker_client_cache[ch_dsn] = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            secure=secure
        )
    return _worker_client_cache[ch_dsn]


def _process_symbol_labels(args: tuple) -> list:
    ch_dsn, symbol, start_dt, end_dt = args

    client = _get_cached_client(ch_dsn)

    query = """
            SELECT toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
                   argMin(open, open_time) AS open,
        max(high) AS high,
        min(low) AS low,
        argMax(close, open_time) AS close,
        sum(volume) AS volume
            FROM bybit.candles
            WHERE symbol = %(symbol)s
              AND interval = 1
              AND open_time >= %(start)s
              AND open_time
                < %(end)s
            GROUP BY bucket
            ORDER BY bucket \
            """

    result = client.query(query, parameters={
        "symbol": symbol,
        "start": start_dt,
        "end": end_dt + timedelta(minutes=15)
    })

    if not result.result_rows:
        return []

    df = pd.DataFrame(
        result.result_rows,
        columns=["bucket", "open", "high", "low", "close", "volume"]
    )
    df["bucket"] = pd.to_datetime(df["bucket"])
    df.set_index("bucket", inplace=True)

    labeler = PumpStartLabelerLookahead()
    df = labeler.detect(df)

    labeled = df[df['pump_start_type'].notna()]
    if labeled.empty:
        return []

    labels = []
    for idx in labeled.index:
        row = labeled.loc[idx]
        labels.append({
            'symbol': symbol,
            'event_open_time': row['start_open_time'],
            'peak_open_time': row['peak_open_time'],
            'pump_la_type': row['pump_start_type'],
            'runup_pct': round(row['pump_start_runup'] * 100, 2)
        })

    return labels


def export_pump_long_labels(ch_dsn: str, start_dt: datetime, end_dt: datetime, max_workers: int = 4) -> pd.DataFrame:
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

    symbols_query = """
                    SELECT DISTINCT symbol
                    FROM bybit.candles
                    WHERE open_time >= %(start_date)s
                      AND open_time <= %(end_date)s
                      AND interval = 1
                    ORDER BY symbol \
                    """
    result = client.query(symbols_query, parameters={
        "start_date": start_dt,
        "end_date": end_dt
    })
    symbols = [row[0] for row in result.result_rows]

    if not symbols:
        return pd.DataFrame()

    log("INFO", "EXPORT", f"found {len(symbols)} symbols")

    tasks = [(ch_dsn, symbol, start_dt, end_dt) for symbol in symbols]

    all_labels = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for labels in executor.map(_process_symbol_labels, tasks):
            all_labels.extend(labels)

    if all_labels:
        df = pd.DataFrame(all_labels)
        df = df.sort_values(['symbol', 'event_open_time']).reset_index(drop=True)
        return df
    return pd.DataFrame()


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Export pump long labels from ClickHouse")

    parser.add_argument("--clickhouse-dsn", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = parse_date_exclusive(args.end_date)

    log("INFO", "EXPORT", f"exporting labels from {args.start_date} to {args.end_date}")

    labels_df = export_pump_long_labels(
        args.clickhouse_dsn,
        start_date,
        end_date,
        max_workers=args.workers
    )

    log("INFO", "EXPORT", f"found {len(labels_df)} labels")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df.to_csv(out_path, index=False)

    config_path = out_path.parent / "run_config_labels.json"
    config = {
        'clickhouse_dsn': args.clickhouse_dsn,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'workers': args.workers,
        'output_path': str(out_path),
        'total_labels': len(labels_df),
        'created_at': datetime.now().isoformat()
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    log("INFO", "EXPORT", f"labels saved to {out_path}")
    log("INFO", "EXPORT", f"config saved to {config_path}")


if __name__ == "__main__":
    main()
