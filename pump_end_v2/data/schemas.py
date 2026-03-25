from __future__ import annotations

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("symbol", "open_time", "open", "high", "low", "close", "volume")


def prepare_ohlcv_15m_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy(deep=True)
    _require_columns(prepared)
    prepared["open_time"] = pd.to_datetime(prepared["open_time"], utc=True, errors="raise")
    prepared = prepared.sort_values(["symbol", "open_time"], kind="mergesort").reset_index(drop=True)
    validate_ohlcv_15m_frame(prepared)
    return prepared


def validate_ohlcv_15m_frame(df: pd.DataFrame) -> None:
    _require_columns(df)
    if df.empty:
        return
    if not isinstance(df["open_time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("open_time must be timezone-aware datetime")
    if str(df["open_time"].dt.tz) != "UTC":
        raise ValueError("open_time timezone must be UTC")
    sorted_df = df.sort_values(["symbol", "open_time"], kind="mergesort").reset_index(drop=True)
    if not df.reset_index(drop=True).equals(sorted_df):
        raise ValueError("frame must be sorted by symbol, open_time")
    duplicates = df.duplicated(subset=["symbol", "open_time"], keep=False)
    if duplicates.any():
        raise ValueError("open_time must be unique within symbol")
    open_time = df["open_time"]
    on_grid = (
        (open_time.dt.minute % 15 == 0)
        & (open_time.dt.second == 0)
        & (open_time.dt.microsecond == 0)
        & (open_time.dt.nanosecond == 0)
    )
    if not on_grid.all():
        raise ValueError("open_time must be aligned to 15m grid")
    if not (df["open"] <= df["high"]).all():
        raise ValueError("open must be <= high")
    if not (df["low"] <= df["high"]).all():
        raise ValueError("low must be <= high")
    if not ((df["low"] <= df["close"]) & (df["close"] <= df["high"])).all():
        raise ValueError("close must be within [low, high]")
    if not ((df["low"] <= df["open"]) & (df["open"] <= df["high"])).all():
        raise ValueError("open must be within [low, high]")
    if not (df["volume"] >= 0).all():
        raise ValueError("volume must be non-negative")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLCV columns: {missing}")
