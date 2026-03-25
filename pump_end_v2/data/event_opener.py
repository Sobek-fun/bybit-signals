from __future__ import annotations

import pandas as pd

from pump_end_v2.config import EventOpenerConfig
from pump_end_v2.logging import stage_done, stage_start


def open_causal_pump_episodes(df: pd.DataFrame, config: EventOpenerConfig) -> pd.DataFrame:
    stage_start("EVENT", "EVENT")
    episodes: list[dict[str, object]] = []
    for symbol, sdf in df.groupby("symbol", sort=False):
        episodes.extend(_open_symbol_episodes(symbol, sdf.reset_index(drop=True), config))
    episodes_df = pd.DataFrame(
        episodes,
        columns=[
            "episode_id",
            "symbol",
            "episode_open_time",
            "episode_close_time",
            "expiry_reason",
            "duration_bars",
            "episode_open_close",
            "max_high_during_episode",
            "max_runup_pct_during_episode",
        ],
    )
    stage_done(
        "EVENT",
        f"EVENT symbols={df['symbol'].nunique()} bars={len(df)} episodes_total={len(episodes_df)}",
    )
    return episodes_df


def _open_symbol_episodes(symbol: str, sdf: pd.DataFrame, config: EventOpenerConfig) -> list[dict[str, object]]:
    rows = _build_context_rows(sdf, config)
    result: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    next_open_idx = 0
    for i, row in rows.iterrows():
        if active is None:
            if i >= next_open_idx and bool(row["pump_context_flag"]):
                open_time = row["open_time"]
                active = {
                    "episode_id": f"{symbol}|{open_time:%Y%m%d_%H%M%S}",
                    "symbol": symbol,
                    "episode_open_time": open_time,
                    "episode_open_close": float(row["close"]),
                    "max_high_during_episode": float(row["high"]),
                    "max_runup_pct_during_episode": float(row["runup_pct"]),
                    "duration_bars": 1,
                }
                expiry_reason = _resolve_expiry_reason(active, row, config)
                if expiry_reason is not None:
                    result.append(_close_episode(active, row["open_time"], expiry_reason))
                    active = None
                    next_open_idx = i + config.cooldown_bars + 1
            continue
        active["duration_bars"] = int(active["duration_bars"]) + 1
        active["max_high_during_episode"] = max(float(active["max_high_during_episode"]), float(row["high"]))
        active["max_runup_pct_during_episode"] = max(float(active["max_runup_pct_during_episode"]), float(row["runup_pct"]))
        expiry_reason = _resolve_expiry_reason(active, row, config)
        if expiry_reason is not None:
            result.append(_close_episode(active, row["open_time"], expiry_reason))
            active = None
            next_open_idx = i + config.cooldown_bars + 1
    if active is not None:
        close_time = rows["open_time"].iloc[-1]
        result.append(_close_episode(active, close_time, "data_end"))
    return result


def _build_context_rows(sdf: pd.DataFrame, config: EventOpenerConfig) -> pd.DataFrame:
    frame = sdf.copy()
    frame["rolling_min_low"] = frame["low"].rolling(window=config.runup_lookback_bars, min_periods=1).min()
    frame["runup_pct"] = frame["close"] / frame["rolling_min_low"] - 1.0
    frame["rolling_max_high"] = frame["high"].rolling(window=config.near_high_lookback_bars, min_periods=1).max()
    frame["near_high_flag"] = frame["close"] >= frame["rolling_max_high"] * (1.0 - config.near_high_tol_pct)
    frame["rolling_mean_volume"] = frame["volume"].rolling(window=config.volume_ratio_lookback_bars, min_periods=1).mean()
    frame["volume_ratio"] = frame["volume"] / frame["rolling_mean_volume"]
    frame["volume_ratio"] = frame["volume_ratio"].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    frame["pump_context_flag"] = (
        (frame["runup_pct"] >= config.min_runup_pct)
        & frame["near_high_flag"]
        & (frame["volume_ratio"] >= config.min_volume_ratio)
    )
    return frame


def _resolve_expiry_reason(active: dict[str, object], row: pd.Series, config: EventOpenerConfig) -> str | None:
    episode_high_so_far = max(float(active["max_high_during_episode"]), float(row["high"]))
    drawdown_pct = (episode_high_so_far - float(row["close"])) / episode_high_so_far
    if int(active["duration_bars"]) >= config.max_episode_bars:
        return "max_age"
    if drawdown_pct >= config.expiry_drawdown_pct:
        return "drawdown"
    return None


def _close_episode(active: dict[str, object], close_time: pd.Timestamp, reason: str) -> dict[str, object]:
    payload = dict(active)
    payload["episode_close_time"] = close_time
    payload["expiry_reason"] = reason
    return payload
