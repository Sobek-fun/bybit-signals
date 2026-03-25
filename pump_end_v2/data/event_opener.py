from __future__ import annotations

import time

import pandas as pd

from pump_end_v2.config import EventOpenerConfig
from pump_end_v2.logging import log_info, stage_done, stage_start


def open_causal_pump_episodes(token_state_df: pd.DataFrame, config: EventOpenerConfig) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("EVENT", "OPEN_EPISODES")
    episodes: list[dict[str, object]] = []
    for symbol, sdf in token_state_df.groupby("symbol", sort=False):
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
    log_info(
        "EVENT",
        f"summary symbols={token_state_df['symbol'].nunique() if not token_state_df.empty else 0} bars={len(token_state_df)} episodes_total={len(episodes_df)}",
    )
    stage_done("EVENT", "OPEN_EPISODES", elapsed_sec=time.perf_counter() - started)
    return episodes_df


def _open_symbol_episodes(symbol: str, sdf: pd.DataFrame, config: EventOpenerConfig) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    next_open_idx = 0
    for i, row in sdf.iterrows():
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
        close_time = sdf["open_time"].iloc[-1]
        result.append(_close_episode(active, close_time, "data_end"))
    return result


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
