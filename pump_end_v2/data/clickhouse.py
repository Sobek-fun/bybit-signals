from datetime import datetime
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd

from pump_end_v2.config import V2Config

_OHLCV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_OHLCV_COLUMNS))


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized.endswith("USDT"):
        normalized = f"{normalized}USDT"
    return normalized


def _to_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_dsn(dsn: str) -> dict[str, object]:
    parsed = urlparse(dsn)
    if not parsed.hostname:
        raise ValueError("invalid ClickHouse DSN: host is required")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 8123,
        "username": parsed.username or "default",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/") if parsed.path else "default",
        "secure": parsed.scheme == "https",
    }


class MinuteOneSecondFetcher:
    def __init__(self, client: object, transactions_table: str) -> None:
        self._client = client
        self._transactions_table = transactions_table
        self._cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}

    def fetch(self, symbol: str, minute_start: datetime | pd.Timestamp) -> pd.DataFrame:
        normalized_symbol = _normalize_symbol(symbol)
        minute_ts = _to_utc_timestamp(minute_start).floor("min")
        cache_key = (normalized_symbol, minute_ts)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        minute_end = minute_ts + pd.Timedelta(minutes=1)
        query = f"""
        SELECT
            symbol,
            toStartOfSecond(transaction_time) AS open_time,
            argMin(price, transaction_time) AS open,
            max(price) AS high,
            min(price) AS low,
            argMax(price, transaction_time) AS close,
            sum(size) AS volume
        FROM {self._transactions_table}
        WHERE symbol = %(symbol)s
          AND transaction_time >= %(start)s
          AND transaction_time < %(end)s
        GROUP BY symbol, open_time
        ORDER BY open_time
        """
        result = self._client.query(
            query,
            parameters={
                "symbol": normalized_symbol,
                "start": minute_ts.to_pydatetime(),
                "end": minute_end.to_pydatetime(),
            },
        )
        if not result.result_rows:
            frame = _empty_ohlcv()
        else:
            frame = pd.DataFrame(result.result_rows, columns=list(_OHLCV_COLUMNS))
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        self._cache[cache_key] = frame
        return frame


class ClickHouseMarketDataLoader:
    def __init__(self, clickhouse_dsn: str, config: V2Config) -> None:
        dsn_fields = _parse_dsn(clickhouse_dsn)
        self._client = clickhouse_connect.get_client(**dsn_fields)
        self._config = config
        self._candles_table = config.data_clickhouse.candles_table
        self._transactions_table = config.data_clickhouse.transactions_table
        self._timezone = config.data_clickhouse.timezone.upper()
        if self._timezone != "UTC":
            raise ValueError("data.clickhouse.timezone must be UTC")

    def build_universe_symbols(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for symbol in self._config.data_universe.symbols:
            normalized = _normalize_symbol(symbol)
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        for symbol in (
            self._config.references.btc_symbol,
            self._config.references.eth_symbol,
        ):
            normalized = _normalize_symbol(symbol)
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return tuple(ordered)

    def load_15m_ohlcv(self) -> pd.DataFrame:
        symbols = self.build_universe_symbols()
        if not symbols:
            return _empty_ohlcv()
        opener = self._config.event_opener
        resolver = self._config.resolver
        execution = self._config.execution
        warmup_bars = (
            max(
                26,
                14,
                12,
                int(opener.runup_lookback_bars),
                int(opener.near_high_lookback_bars),
                int(opener.volume_ratio_lookback_bars),
            )
            + 2
        )
        tail_15m_bars = max(
            int(resolver.horizon_bars),
            int(execution.entry_shift_bars) + int(execution.max_hold_bars),
        )
        start_anchor = _to_utc_timestamp(self._config.data_window.start)
        end_anchor_raw = (
            self._config.data_window.end
            if self._config.data_window.end is not None
            else self._config.splits.test_end
        )
        end_anchor = _to_utc_timestamp(end_anchor_raw)
        query_start = start_anchor - pd.Timedelta(minutes=15 * warmup_bars)
        query_end = end_anchor + pd.Timedelta(minutes=15 * (tail_15m_bars + 1))
        query = f"""
        SELECT
            symbol,
            toStartOfInterval(open_time, INTERVAL 15 minute) AS bucket_start,
            argMin(open, open_time) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, open_time) AS close,
            sum(volume) AS volume
        FROM {self._candles_table}
        WHERE interval = 1
          AND symbol IN %(symbols)s
          AND open_time >= %(start)s
          AND open_time < %(end)s
        GROUP BY symbol, bucket_start
        ORDER BY symbol, bucket_start
        """
        result = self._client.query(
            query,
            parameters={
                "symbols": list(symbols),
                "start": query_start.to_pydatetime(),
                "end": query_end.to_pydatetime(),
            },
        )
        if not result.result_rows:
            return _empty_ohlcv()
        frame = pd.DataFrame(
            result.result_rows,
            columns=["symbol", "bucket_start", "open", "high", "low", "close", "volume"],
        )
        frame = frame.rename(columns={"bucket_start": "open_time"})
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame

    def load_1m_ohlcv(
        self,
        symbols: tuple[str, ...] | list[str],
        start_time: datetime | pd.Timestamp,
        end_time: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        normalized_symbols = tuple(
            _normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()
        )
        if not normalized_symbols:
            return _empty_ohlcv()
        start_ts = _to_utc_timestamp(start_time)
        end_ts = _to_utc_timestamp(end_time)
        if end_ts <= start_ts:
            return _empty_ohlcv()
        query = f"""
        SELECT
            symbol,
            open_time,
            open,
            high,
            low,
            close,
            volume
        FROM {self._candles_table}
        WHERE interval = 1
          AND symbol IN %(symbols)s
          AND open_time >= %(start)s
          AND open_time < %(end)s
        ORDER BY symbol, open_time
        """
        result = self._client.query(
            query,
            parameters={
                "symbols": list(normalized_symbols),
                "start": start_ts.to_pydatetime(),
                "end": end_ts.to_pydatetime(),
            },
        )
        if not result.result_rows:
            return _empty_ohlcv()
        frame = pd.DataFrame(result.result_rows, columns=list(_OHLCV_COLUMNS))
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame

    def load_1m_ohlcv_intervals(
        self,
        symbol_intervals: dict[
            str, list[tuple[datetime | pd.Timestamp, datetime | pd.Timestamp]]
        ],
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for symbol, intervals in symbol_intervals.items():
            normalized_symbol = _normalize_symbol(symbol)
            for start_time, end_time in intervals:
                chunk = self.load_1m_ohlcv(
                    symbols=(normalized_symbol,),
                    start_time=start_time,
                    end_time=end_time,
                )
                if not chunk.empty:
                    chunks.append(chunk)
        if not chunks:
            return _empty_ohlcv()
        merged = pd.concat(chunks, ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "open_time"], keep="first")
        merged = merged.sort_values(["symbol", "open_time"], kind="mergesort").reset_index(
            drop=True
        )
        return merged

    def build_1s_fetcher(self) -> MinuteOneSecondFetcher:
        return MinuteOneSecondFetcher(
            client=self._client,
            transactions_table=self._transactions_table,
        )
