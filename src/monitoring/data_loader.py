import pandas as pd
from datetime import datetime, timedelta
import clickhouse_connect
from urllib.parse import urlparse


class DataLoader:
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

    def load_candles(self, symbol: str, lookback: int) -> pd.DataFrame:
        t = self._get_last_closed_time(symbol)
        if t is None:
            return pd.DataFrame()

        start_time = t - timedelta(minutes=lookback * 15)

        query = """
        SELECT
            toStartOfInterval(transaction_time, INTERVAL 15 minute) AS bucket,
            argMin(price, transaction_time) AS open,
            max(price) AS high,
            min(price) AS low,
            argMax(price, transaction_time) AS close,
            sum(size) AS volume,
            sumIf(size, side = 'Buy') AS buy_volume,
            sumIf(size, side = 'Sell') AS sell_volume,
            sumIf(size, side = 'Buy') - sumIf(size, side = 'Sell') AS net_volume,
            count() AS trades_count
        FROM bybit.transactions
        WHERE symbol = %(symbol)s
          AND bucket >= %(start)s
          AND bucket <= %(end)s
        GROUP BY bucket
        ORDER BY bucket
        """

        result = self.client.query(query, parameters={
            "symbol": symbol,
            "start": start_time,
            "end": t
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
        SELECT toStartOfInterval(transaction_time, INTERVAL 15 minute) AS bucket
        FROM bybit.transactions
        WHERE symbol = %(symbol)s
          AND transaction_time < %(current_bucket_start)s
        ORDER BY bucket DESC
        LIMIT 1
        """

        result = self.client.query(query, parameters={
            "symbol": symbol,
            "current_bucket_start": current_bucket_start
        })

        if not result.result_rows:
            return None

        return result.result_rows[0][0]
