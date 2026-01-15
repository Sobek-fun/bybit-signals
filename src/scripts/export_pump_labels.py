import argparse
from datetime import datetime
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from src.monitoring.data_loader import DataLoader
from src.monitoring.pump_labeler_lookahead import PumpLabelerLookahead


def main():
    parser = argparse.ArgumentParser(description="Export pump labels for all symbols")
    parser.add_argument(
        "--clickhouse-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN (e.g., http://user:pass@host:port/database)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pump_labels.csv",
        help="Output CSV file path (default: pump_labels.csv)"
    )

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d %H:%M:%S')

    parsed = urlparse(args.clickhouse_dsn)
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

    print(f"Fetching symbols from {args.start_date} to {args.end_date}...")
    symbols_query = """
    SELECT DISTINCT symbol
    FROM bybit.candles
    WHERE open_time >= %(start_date)s
      AND open_time <= %(end_date)s
      AND interval = 1
    ORDER BY symbol
    """
    result = client.query(symbols_query, parameters={
        "start_date": start_dt,
        "end_date": end_dt
    })
    symbols = [row[0] for row in result.result_rows]

    if not symbols:
        print("No symbols found for the specified date range")
        return

    print(f"Found {len(symbols)} symbols")

    loader = DataLoader(args.clickhouse_dsn)
    labeler = PumpLabelerLookahead()

    all_labels = []

    for symbol in symbols:
        print(f"Processing {symbol}...")

        df = loader.load_candles_range(symbol, start_dt, end_dt)

        if df.empty:
            print(f"  No data for {symbol}")
            continue

        df = labeler.detect(df)

        labeled = df[df['pump_la_type'].notna()]

        if not labeled.empty:
            for _, row in labeled.iterrows():
                all_labels.append({
                    'symbol': symbol,
                    'event_open_time': row['event_open_time'],
                    'peak_open_time': row['peak_open_time'],
                    'pump_la_type': row['pump_la_type'],
                    'runup_pct': round(row['pump_la_runup'] * 100, 2)
                })
            print(
                f"  Found {len(labeled)} labels (A={len(labeled[labeled['pump_la_type'] == 'A'])}, B={len(labeled[labeled['pump_la_type'] == 'B'])})")
        else:
            print(f"  No labels found")

    if all_labels:
        labels_df = pd.DataFrame(all_labels)
        labels_df.to_csv(args.output, index=False)
        print(f"\nExported {len(all_labels)} labels to {args.output}")
    else:
        print("\nNo labels found across all symbols")


if __name__ == "__main__":
    main()
