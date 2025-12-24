from datetime import datetime, timedelta
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


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
        SELECT
            symbol,
            toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
            argMin(open, open_time) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, open_time) AS close,
            sum(volume) AS volume,
            0 AS buy_volume,
            0 AS sell_volume,
            0 AS net_volume,
            0 AS trades_count
        FROM bybit.candles
        WHERE symbol IN %(symbols)s
          AND interval = 1
          AND open_time >= %(start)s
          AND open_time < %(end)s
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

        if query_duration_ms > self.SLOW_QUERY_THRESHOLD_MS:
            log("WARN", "DATA",
                f"batch query slow: {query_duration_ms:.0f}ms rows={len(result.result_rows)} symbols={len(symbols)}")

        if not result.result_rows:
            return {}

        df_all = pd.DataFrame(
            result.result_rows,
            columns=["symbol", "bucket", "open", "high", "low", "close", "volume",
                     "buy_volume", "sell_volume", "net_volume", "trades_count"]
        )

        df_all["bucket"] = pd.to_datetime(df_all["bucket"])

        result_dict = {}
        for symbol in df_all["symbol"].unique():
            symbol_df = df_all[df_all["symbol"] == symbol].copy()
            symbol_df = symbol_df.drop(columns=["symbol"])
            symbol_df.set_index("bucket", inplace=True)
            result_dict[symbol] = symbol_df

        return result_dict

    def load_candles(self, symbol: str, lookback: int) -> pd.DataFrame:
        t = self._get_last_closed_time(symbol)
        if t is None:
            log("WARN", "DATA", f"symbol={symbol} no closed candles found")
            return pd.DataFrame()

        start_time = t - timedelta(minutes=lookback * 15)

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
          AND open_time <= %(end)s
        ORDER BY open_time
        """

        query_start = datetime.now()
        result = self.client.query(query, parameters={
            "symbol": symbol,
            "start": start_time,
            "end": t
        })
        query_duration_ms = (datetime.now() - query_start).total_seconds() * 1000

        if query_duration_ms > self.SLOW_QUERY_THRESHOLD_MS:
            log("WARN", "DATA", f"symbol={symbol} slow query: {query_duration_ms:.0f}ms rows={len(result.result_rows)}")

        if not result.result_rows:
            log("WARN", "DATA", f"symbol={symbol} returned 0 rows for lookback={lookback}")
            return pd.DataFrame()

        df = pd.DataFrame(
            result.result_rows,
            columns=["bucket", "open", "high", "low", "close", "volume",
                     "buy_volume", "sell_volume", "net_volume", "trades_count"]
        )

        df["bucket"] = pd.to_datetime(df["bucket"])
        df.set_index("bucket", inplace=True)

        return df

    def load_candles_range(self, symbol: str, start_bucket: datetime, end_bucket: datetime) -> pd.DataFrame:
        query = """
        SELECT
            toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
            argMin(open, open_time) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, open_time) AS close,
            sum(volume) AS volume,
            0 AS buy_volume,
            0 AS sell_volume,
            0 AS net_volume,
            0 AS trades_count
        FROM bybit.candles
        WHERE symbol = %(symbol)s
          AND interval = 1
          AND open_time >= %(start)s
          AND open_time < %(end)s
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
            columns=["bucket", "open", "high", "low", "close", "volume",
                     "buy_volume", "sell_volume", "net_volume", "trades_count"]
        )

        df["bucket"] = pd.to_datetime(df["bucket"])
        df.set_index("bucket", inplace=True)

        return df

    def _get_last_closed_time(self, symbol: str) -> datetime | None:
        now = datetime.now()
        minutes = (now.minute // 15) * 15
        current_bucket_start = now.replace(minute=minutes, second=0, microsecond=0)

        if now.minute % 15 == 0 and now.second < self.offset_seconds:
            current_bucket_start = current_bucket_start - timedelta(minutes=15)

        query = """
        SELECT open_time
        FROM bybit.candles
        WHERE symbol = %(symbol)s
          AND interval = 1
          AND open_time < %(current_bucket_start)s
        ORDER BY open_time DESC
        LIMIT 1
        """

        result = self.client.query(query, parameters={
            "symbol": symbol,
            "current_bucket_start": current_bucket_start
        })

        if not result.result_rows:
            return None

        return result.result_rows[0][0]
