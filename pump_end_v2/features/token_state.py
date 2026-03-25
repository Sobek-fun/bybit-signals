from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pump_end_v2.config import EventOpenerConfig
from pump_end_v2.logging import log_info, stage_done, stage_start

TOKEN_STATE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_ret_1",
    "close_ret_4",
    "close_ret_12",
    "intrabar_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "rolling_volatility_4",
    "rolling_volatility_12",
    "runup_pct",
    "recent_high_distance_pct",
    "near_high_flag",
    "volume_ratio",
    "pump_context_flag",
)


def build_token_state_layer(df: pd.DataFrame, event_opener_config: EventOpenerConfig) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("LAYERS", "TOKEN_STATE")
    if df.empty:
        out = pd.DataFrame(columns=list(TOKEN_STATE_COLUMNS))
        log_info(
            "LAYERS",
            "token_state summary symbols=0 rows_total=0 cols_total=21",
        )
        stage_done("LAYERS", "TOKEN_STATE", elapsed_sec=time.perf_counter() - started)
        return out
    parts = []
    for symbol, sdf in df.groupby("symbol", sort=False):
        parts.append(_build_symbol_token_state(symbol, sdf.reset_index(drop=True), event_opener_config))
    token_state = pd.concat(parts, ignore_index=True)
    token_state = token_state.loc[:, list(TOKEN_STATE_COLUMNS)].copy()
    log_info(
        "LAYERS",
        f"token_state summary symbols={token_state['symbol'].nunique()} rows_total={len(token_state)} cols_total={len(token_state.columns)}",
    )
    stage_done("LAYERS", "TOKEN_STATE", elapsed_sec=time.perf_counter() - started)
    return token_state


def _build_symbol_token_state(symbol: str, sdf: pd.DataFrame, cfg: EventOpenerConfig) -> pd.DataFrame:
    frame = sdf.copy()
    frame["symbol"] = symbol
    prev_close = frame["close"].shift(1)
    frame["close_ret_1"] = _safe_ratio(frame["close"], prev_close) - 1.0
    frame["close_ret_4"] = _safe_ratio(frame["close"], frame["close"].shift(4)) - 1.0
    frame["close_ret_12"] = _safe_ratio(frame["close"], frame["close"].shift(12)) - 1.0
    frame["intrabar_range_pct"] = _safe_ratio(frame["high"] - frame["low"], frame["open"])
    frame["candle_body_pct"] = _safe_ratio(frame["close"] - frame["open"], frame["open"])
    frame["upper_wick_pct"] = _safe_ratio(frame["high"] - frame[["open", "close"]].max(axis=1), frame["open"])
    frame["lower_wick_pct"] = _safe_ratio(frame[["open", "close"]].min(axis=1) - frame["low"], frame["open"])
    frame["rolling_volatility_4"] = frame["close_ret_1"].rolling(window=4, min_periods=1).std(ddof=0)
    frame["rolling_volatility_12"] = frame["close_ret_1"].rolling(window=12, min_periods=1).std(ddof=0)
    frame["rolling_min_low"] = frame["low"].rolling(window=cfg.runup_lookback_bars, min_periods=1).min()
    frame["runup_pct"] = _safe_ratio(frame["close"], frame["rolling_min_low"]) - 1.0
    frame["rolling_max_high"] = frame["high"].rolling(window=cfg.near_high_lookback_bars, min_periods=1).max()
    frame["recent_high_distance_pct"] = _safe_ratio(frame["rolling_max_high"] - frame["close"], frame["rolling_max_high"])
    frame["near_high_flag"] = frame["recent_high_distance_pct"] <= cfg.near_high_tol_pct
    frame["rolling_mean_volume"] = frame["volume"].rolling(window=cfg.volume_ratio_lookback_bars, min_periods=1).mean()
    frame["volume_ratio"] = _safe_ratio(frame["volume"], frame["rolling_mean_volume"])
    frame["pump_context_flag"] = (
        (frame["runup_pct"] >= cfg.min_runup_pct)
        & frame["near_high_flag"].fillna(False)
        & (frame["volume_ratio"] >= cfg.min_volume_ratio)
    )
    frame["near_high_flag"] = frame["near_high_flag"].fillna(False).astype(bool)
    frame["pump_context_flag"] = frame["pump_context_flag"].fillna(False).astype(bool)
    return frame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    den = den.where(den > 0.0)
    return (num / den).replace([np.inf, -np.inf], np.nan)
