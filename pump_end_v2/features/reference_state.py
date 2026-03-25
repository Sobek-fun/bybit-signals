from __future__ import annotations

import time

import pandas as pd

from pump_end_v2.logging import log_info, stage_done, stage_start

REFERENCE_STATE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "btc_close_ret_1",
    "btc_close_ret_4",
    "btc_close_ret_12",
    "btc_intrabar_range_pct",
    "btc_volume_ratio",
    "btc_pump_context_flag",
    "eth_close_ret_1",
    "eth_close_ret_4",
    "eth_close_ret_12",
    "eth_intrabar_range_pct",
    "eth_volume_ratio",
    "eth_pump_context_flag",
)


def build_reference_state_layer(token_state_df: pd.DataFrame, btc_symbol: str, eth_symbol: str) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("LAYERS", "REFERENCE_STATE")
    _require_symbol(token_state_df, btc_symbol, "btc_symbol")
    _require_symbol(token_state_df, eth_symbol, "eth_symbol")
    base_times = (
        token_state_df[["open_time"]]
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time", kind="mergesort")
        .reset_index(drop=True)
    )
    btc = _select_symbol(token_state_df, btc_symbol, "btc")
    eth = _select_symbol(token_state_df, eth_symbol, "eth")
    reference = base_times.merge(btc, on="open_time", how="left").merge(eth, on="open_time", how="left")
    reference = reference.loc[:, list(REFERENCE_STATE_COLUMNS)].copy()
    log_info(
        "LAYERS",
        f"reference_state summary rows_total={len(reference)} cols_total={len(reference.columns)}",
    )
    stage_done("LAYERS", "REFERENCE_STATE", elapsed_sec=time.perf_counter() - started)
    return reference


def _require_symbol(token_state_df: pd.DataFrame, symbol: str, field_name: str) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if token_state_df[token_state_df["symbol"] == symbol].empty:
        raise ValueError(f"reference symbol not found in token_state_df: {symbol}")


def _select_symbol(token_state_df: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    cols = {
        "close_ret_1": f"{prefix}_close_ret_1",
        "close_ret_4": f"{prefix}_close_ret_4",
        "close_ret_12": f"{prefix}_close_ret_12",
        "intrabar_range_pct": f"{prefix}_intrabar_range_pct",
        "volume_ratio": f"{prefix}_volume_ratio",
        "pump_context_flag": f"{prefix}_pump_context_flag",
    }
    selected = token_state_df[token_state_df["symbol"] == symbol][["open_time", *cols.keys()]].copy()
    return selected.rename(columns=cols)
