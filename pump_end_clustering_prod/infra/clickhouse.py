from datetime import datetime, timedelta
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from pump_end_clustering_prod.infra.logging import log


class DataLoader:
    SLOW_QUERY_THRESHOLD_MS = 1500

    def __init__(self, ch_dsn: str, offset_seconds: int = 10):
        self.client = self._create_client(ch_dsn)
        self.offset_seconds = offset_seconds

    def _create_client(self, dsn: str):
        parsed = urlparse(dsn)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8123
        username = parsed.username or "default"
        password = parsed.password or ""
        database = parsed.path.lstrip("/") if parsed.path else "default"
        secure = parsed.scheme == "https"

        return clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            secure=secure
        )

    def load_candles_batch(self, symbols: list[str], start_bucket: datetime, end_bucket: datetime) -> dict[
        str, pd.DataFrame]:
        query = """
                SELECT symbol,
                       toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
                       argMin(open, open_time) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, open_time) AS close,
            sum(volume) AS volume
                FROM bybit.candles
                WHERE symbol IN %(symbols)s
                  AND interval = 1
                  AND open_time >= %(start)s
                  AND open_time
                    < %(end)s
                GROUP BY symbol, bucket
                ORDER BY symbol, bucket
                """

        query_start = datetime.now()
        result = self.client.query(query, parameters={
            "symbols": symbols,
            "start": start_bucket,
            "end": end_bucket + timedelta(minutes=15)
        })
        query_duration_ms = (datetime.now() - query_start).total_seconds() * 1000

        total_rows = len(result.result_rows)
        rows_per_symbol = total_rows / len(symbols) if symbols else 0

        log_level = "WARN" if query_duration_ms > self.SLOW_QUERY_THRESHOLD_MS else "INFO"
        log(log_level, "DATA",
            f"batch load symbols={len(symbols)} rows={total_rows} rows_per_symbol={rows_per_symbol:.0f} query_ms={query_duration_ms:.0f}")

        if not result.result_rows:
            return {}

        df_all = pd.DataFrame(
            result.result_rows,
            columns=["symbol", "bucket", "open", "high", "low", "close", "volume"]
        )

        df_all["bucket"] = pd.to_datetime(df_all["bucket"])

        result_dict = {}
        for symbol, group in df_all.groupby("symbol", sort=False):
            symbol_df = group.drop(columns=["symbol"]).set_index("bucket")
            result_dict[symbol] = symbol_df

        return result_dict

    def load_candles_range(self, symbol: str, start_bucket: datetime, end_bucket: datetime) -> pd.DataFrame:
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
                ORDER BY bucket
                """

        result = self.client.query(query, parameters={
            "symbol": symbol,
            "start": start_bucket,
            "end": end_bucket + timedelta(minutes=15)
        })

        if not result.result_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            result.result_rows,
            columns=["bucket", "open", "high", "low", "close", "volume"]
        )

        df["bucket"] = pd.to_datetime(df["bucket"])
        df.set_index("bucket", inplace=True)

        return df


def list_all_usdt_tokens(ch_dsn: str) -> list[str]:
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

    query = "SELECT DISTINCT symbol FROM bybit.transactions WHERE endsWith(symbol, 'USDT') ORDER BY symbol"
    result = client.query(query)
    return [row[0][:-4] for row in result.result_rows]
