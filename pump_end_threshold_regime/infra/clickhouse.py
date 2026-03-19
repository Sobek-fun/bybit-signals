from datetime import datetime, timedelta
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from pump_end_threshold_regime.infra.logging import log


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
            sum(volume) AS volume
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
            columns=["symbol", "bucket", "open", "high", "low", "close", "volume"]
        )

        df_all["bucket"] = pd.to_datetime(df_all["bucket"])

        result_dict = {}
        for symbol, group in df_all.groupby("symbol", sort=False):
            symbol_df = group.drop(columns=["symbol"]).set_index("bucket")
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
            volume
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
            columns=["bucket", "open", "high", "low", "close", "volume"]
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
            sum(volume) AS volume
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
            columns=["bucket", "open", "high", "low", "close", "volume"]
        )

        df["bucket"] = pd.to_datetime(df["bucket"])
        df.set_index("bucket", inplace=True)

        return df

    def load_raw_1m_candles(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query = """
        SELECT
            open_time,
            open,
            high,
            low,
            close,
            volume
        FROM bybit.candles
        WHERE symbol = %(symbol)s
          AND interval = 1
          AND open_time >= %(start)s
          AND open_time < %(end)s
        ORDER BY open_time
        """

        result = self.client.query(query, parameters={
            "symbol": symbol,
            "start": start_time,
            "end": end_time
        })

        if not result.result_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            result.result_rows,
            columns=["open_time", "open", "high", "low", "close", "volume"]
        )

        df["open_time"] = pd.to_datetime(df["open_time"])
        df.set_index("open_time", inplace=True)

        return df

    def load_transactions_range(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query_template = """
        SELECT
            {time_col},
            price,
            size
        FROM bybit.transactions
        WHERE symbol = %(symbol)s
          AND {time_col} >= %(start)s
          AND {time_col} < %(end)s
        ORDER BY {time_col}
        """

        query_start = datetime.now()
        result = None
        for time_col in ("transaction_time", "timestamp"):
            query = query_template.format(time_col=time_col)
            try:
                result = self.client.query(query, parameters={
                    "symbol": symbol,
                    "start": start_time,
                    "end": end_time
                })
                break
            except Exception as e:
                if "Unknown expression or function identifier" in str(e) and f"`{time_col}`" in str(e):
                    continue
                raise

        if result is None:
            return pd.DataFrame()

        query_duration_ms = (datetime.now() - query_start).total_seconds() * 1000

        if query_duration_ms > self.SLOW_QUERY_THRESHOLD_MS:
            log("WARN", "DATA",
                f"transactions query slow: {query_duration_ms:.0f}ms rows={len(result.result_rows)} symbol={symbol}")

        if not result.result_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            result.result_rows,
            columns=["timestamp", "price", "size"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    def load_1s_bars_from_transactions(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query_template = """
        SELECT
            toStartOfSecond({time_col}) AS second,
            argMin(price, {time_col}) AS open,
            max(price) AS high,
            min(price) AS low,
            argMax(price, {time_col}) AS close,
            sum(size) AS volume,
            count() AS trades_count
        FROM bybit.transactions
        WHERE symbol = %(symbol)s
          AND {time_col} >= %(start)s
          AND {time_col} < %(end)s
        GROUP BY second
        ORDER BY second
        """

        query_start = datetime.now()
        result = None
        for time_col in ("transaction_time", "timestamp"):
            query = query_template.format(time_col=time_col)
            try:
                result = self.client.query(query, parameters={
                    "symbol": symbol,
                    "start": start_time,
                    "end": end_time
                })
                break
            except Exception as e:
                if "Unknown expression or function identifier" in str(e) and f"`{time_col}`" in str(e):
                    continue
                raise

        if result is None:
            return pd.DataFrame()

        query_duration_ms = (datetime.now() - query_start).total_seconds() * 1000

        if query_duration_ms > self.SLOW_QUERY_THRESHOLD_MS:
            log("WARN", "DATA",
                f"1s bars query slow: {query_duration_ms:.0f}ms rows={len(result.result_rows)} symbol={symbol}")

        if not result.result_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            result.result_rows,
            columns=["second", "open", "high", "low", "close", "volume", "trades_count"]
        )

        df["second"] = pd.to_datetime(df["second"])
        df.set_index("second", inplace=True)

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


def get_liquid_universe(ch_dsn: str, start_dt: datetime, end_dt: datetime, top_n: int = 120,
                        exclude: list[str] = None) -> list[str]:
    if exclude is None:
        exclude = ["BTCUSDT", "ETHUSDT"]

    parsed = urlparse(ch_dsn)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8123
    username = parsed.username or "default"
    password = parsed.password or ""
    database = parsed.path.lstrip("/") if parsed.path else "default"
    secure = parsed.scheme == "https"

    client = clickhouse_connect.get_client(
        host=host, port=port, username=username,
        password=password, database=database, secure=secure
    )

    exclude_clause = ""
    if exclude:
        placeholders = ", ".join([f"'{s}'" for s in exclude])
        exclude_clause = f"AND symbol NOT IN ({placeholders})"

    query = f"""
    SELECT symbol, avg(volume * close) AS avg_dollar_volume
    FROM (
        SELECT
            symbol,
            toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket,
            sum(volume) AS volume,
            argMax(close, open_time) AS close
        FROM bybit.candles
        WHERE interval = 1
          AND open_time >= %(start)s
          AND open_time < %(end)s
          AND endsWith(symbol, 'USDT')
          {exclude_clause}
        GROUP BY symbol, bucket
    )
    GROUP BY symbol
    ORDER BY avg_dollar_volume DESC
    LIMIT %(top_n)s
    """

    result = client.query(query, parameters={
        "start": start_dt,
        "end": end_dt,
        "top_n": top_n,
    })

    return [row[0] for row in result.result_rows]


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
