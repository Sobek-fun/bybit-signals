from __future__ import annotations

from typing import Any

import pandas as pd

from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.logging import log_info

_BARS_REQUIRED_COLUMNS: tuple[str, ...] = ("symbol", "open_time", "open", "high", "low", "close")
_DECISION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "gate_decision",
    "gate_block_threshold",
    "p_block",
    "target_block_signal",
    "block_reason",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
)
_EXECUTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "execution_status",
    "entry_price",
    "exit_time",
    "exit_price",
    "trade_outcome",
    "trade_pnl_pct",
    "mfe_pct",
    "mae_pct",
    "holding_bars",
)


def prepare_intraday_bars_frame(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    if timeframe_label not in {"1m", "1s"}:
        raise ValueError("timeframe_label must be one of: 1m, 1s")
    _require_columns(df, _BARS_REQUIRED_COLUMNS, f"bars_{timeframe_label}_df")
    frame = df.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.sort_values(["symbol", "open_time"], kind="mergesort").reset_index(drop=True)
    dup_mask = frame.duplicated(subset=["symbol", "open_time"], keep=False)
    if dup_mask.any():
        sample = frame.loc[dup_mask, ["symbol", "open_time"]].head(3).to_dict("records")
        raise ValueError(f"duplicate open_time within symbol for {timeframe_label}: sample={sample}")
    valid_price_mask = (
        (frame["open"] <= frame["high"])
        & (frame["low"] <= frame["high"])
        & (frame["low"] <= frame["close"])
        & (frame["close"] <= frame["high"])
        & (frame["low"] <= frame["open"])
        & (frame["open"] <= frame["high"])
    )
    if not bool(valid_price_mask.all()):
        raise ValueError(f"invalid OHLC bounds in {timeframe_label} bars")
    return frame


def replay_short_signals_with_symbol_lock(
    decision_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(decision_df, _DECISION_REQUIRED_COLUMNS, "decision_df")
    decisions = decision_df.copy().reset_index(drop=False).rename(columns={"index": "_row_order"})
    decisions["context_bar_open_time"] = pd.to_datetime(decisions["context_bar_open_time"], utc=True, errors="raise")
    decisions["decision_time"] = pd.to_datetime(decisions["decision_time"], utc=True, errors="raise")
    decisions["entry_bar_open_time"] = pd.to_datetime(decisions["entry_bar_open_time"], utc=True, errors="raise")
    decisions["gate_decision"] = decisions["gate_decision"].astype(str).str.lower()
    bars_1m = prepare_intraday_bars_frame(bars_1m_df, "1m")
    bars_1s = prepare_intraday_bars_frame(bars_1s_df, "1s") if bars_1s_df is not None else None
    for column in _EXECUTION_OUTPUT_COLUMNS:
        decisions[column] = pd.NA
    decisions["execution_status"] = "pending"
    blocked_gate_mask = decisions["gate_decision"] == "block"
    decisions.loc[blocked_gate_mask, "execution_status"] = "blocked_gate"
    decisions.loc[blocked_gate_mask, "trade_outcome"] = pd.NA
    lock_until_by_symbol: dict[str, pd.Timestamp] = {}
    kept_indices = decisions.index[decisions["gate_decision"] == "keep"].tolist()
    kept_ordered = decisions.loc[kept_indices].sort_values(["entry_bar_open_time", "signal_id"], kind="mergesort")
    bars_1m_by_symbol = {symbol: grp.reset_index(drop=True) for symbol, grp in bars_1m.groupby("symbol", sort=False)}
    bars_1s_by_symbol = (
        {symbol: grp.reset_index(drop=True) for symbol, grp in bars_1s.groupby("symbol", sort=False)}
        if bars_1s is not None
        else {}
    )
    hold_window_delta = pd.Timedelta(minutes=int(execution_contract.max_hold_bars) * 15)
    for idx, row in kept_ordered.iterrows():
        symbol = str(row["symbol"])
        entry_time = pd.Timestamp(row["entry_bar_open_time"])
        if symbol in lock_until_by_symbol and entry_time < lock_until_by_symbol[symbol]:
            decisions.at[idx, "execution_status"] = "blocked_symbol_lock"
            continue
        symbol_1m = bars_1m_by_symbol.get(symbol)
        if symbol_1m is None:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        entry_match = symbol_1m[symbol_1m["open_time"] == entry_time]
        if entry_match.empty:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        entry_bar = entry_match.iloc[0]
        entry_price = float(entry_bar["open"])
        tp_price = entry_price * (1.0 - float(execution_contract.tp_pct))
        sl_price = entry_price * (1.0 + float(execution_contract.sl_pct))
        horizon_end = entry_time + hold_window_delta
        path_bars = symbol_1m[(symbol_1m["open_time"] >= entry_time) & (symbol_1m["open_time"] < horizon_end)].copy()
        if path_bars.empty:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        outcome_data = _replay_single_short_path(
            symbol=symbol,
            path_1m=path_bars,
            bars_1s=bars_1s_by_symbol.get(symbol),
            tp_price=tp_price,
            sl_price=sl_price,
            entry_price=entry_price,
        )
        decisions.at[idx, "execution_status"] = "executed"
        decisions.at[idx, "entry_price"] = entry_price
        decisions.at[idx, "exit_time"] = outcome_data["exit_time"]
        decisions.at[idx, "exit_price"] = outcome_data["exit_price"]
        decisions.at[idx, "trade_outcome"] = outcome_data["trade_outcome"]
        decisions.at[idx, "trade_pnl_pct"] = ((entry_price - float(outcome_data["exit_price"])) / entry_price) * 100.0
        decisions.at[idx, "mfe_pct"] = outcome_data["mfe_pct"]
        decisions.at[idx, "mae_pct"] = outcome_data["mae_pct"]
        decisions.at[idx, "holding_bars"] = int(outcome_data["holding_bars"])
        lock_until_by_symbol[symbol] = pd.Timestamp(outcome_data["exit_time"])
    decisions = decisions.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])
    executed_signals_df = decisions[decisions["execution_status"] == "executed"].copy().reset_index(drop=True)
    blocked_gate = int((decisions["execution_status"] == "blocked_gate").sum())
    blocked_symbol_lock = int((decisions["execution_status"] == "blocked_symbol_lock").sum())
    log_info(
        "EXECUTION",
        (
            "execution replay done "
            f"decisions_total={len(decisions)} executed_total={len(executed_signals_df)} "
            f"blocked_gate={blocked_gate} blocked_symbol_lock={blocked_symbol_lock}"
        ),
    )
    return decisions.reset_index(drop=True), executed_signals_df


def _replay_single_short_path(
    symbol: str,
    path_1m: pd.DataFrame,
    bars_1s: pd.DataFrame | None,
    tp_price: float,
    sl_price: float,
    entry_price: float,
) -> dict[str, Any]:
    trade_outcome = "timeout"
    exit_time = pd.Timestamp(path_1m["open_time"].iloc[-1])
    exit_price = float(path_1m["close"].iloc[-1])
    holding_bars = int(len(path_1m))
    processed: list[pd.Series] = []
    for bar in path_1m.itertuples(index=False):
        bar_time = pd.Timestamp(bar.open_time)
        bar_low = float(bar.low)
        bar_high = float(bar.high)
        bar_close = float(bar.close)
        processed.append(pd.Series({"low": bar_low, "high": bar_high}))
        hit_tp = bar_low <= tp_price
        hit_sl = bar_high >= sl_price
        if hit_tp and hit_sl:
            if bars_1s is None:
                trade_outcome = "ambiguous"
                exit_time = bar_time
                exit_price = bar_close
                holding_bars = len(processed)
                break
            second_outcome = _resolve_intraminute_touch(
                minute_start=bar_time,
                bars_1s=bars_1s,
                tp_price=tp_price,
                sl_price=sl_price,
            )
            if second_outcome["trade_outcome"] is None:
                trade_outcome = "ambiguous"
                exit_time = bar_time
                exit_price = bar_close
            else:
                trade_outcome = str(second_outcome["trade_outcome"])
                exit_time = pd.Timestamp(second_outcome["exit_time"])
                exit_price = float(second_outcome["exit_price"])
            holding_bars = len(processed)
            break
        if hit_tp:
            trade_outcome = "tp"
            exit_time = bar_time
            exit_price = tp_price
            holding_bars = len(processed)
            break
        if hit_sl:
            trade_outcome = "sl"
            exit_time = bar_time
            exit_price = sl_price
            holding_bars = len(processed)
            break
    processed_df = pd.DataFrame(processed)
    mfe_pct = 0.0
    mae_pct = 0.0
    if not processed_df.empty:
        mfe_pct = float(((entry_price - processed_df["low"]) / entry_price).max() * 100.0)
        mae_pct = float(((processed_df["high"] - entry_price) / entry_price).max() * 100.0)
    return {
        "symbol": symbol,
        "trade_outcome": trade_outcome,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "holding_bars": holding_bars,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }


def _resolve_intraminute_touch(
    minute_start: pd.Timestamp,
    bars_1s: pd.DataFrame,
    tp_price: float,
    sl_price: float,
) -> dict[str, Any]:
    minute_end = minute_start + pd.Timedelta(minutes=1)
    minute_bars = bars_1s[(bars_1s["open_time"] >= minute_start) & (bars_1s["open_time"] < minute_end)]
    if minute_bars.empty:
        return {"trade_outcome": None, "exit_time": None, "exit_price": None}
    for row in minute_bars.itertuples(index=False):
        sec_low = float(row.low)
        sec_high = float(row.high)
        hit_tp = sec_low <= tp_price
        hit_sl = sec_high >= sl_price
        if hit_tp and hit_sl:
            return {"trade_outcome": "ambiguous", "exit_time": pd.Timestamp(row.open_time), "exit_price": float(row.close)}
        if hit_tp:
            return {"trade_outcome": "tp", "exit_time": pd.Timestamp(row.open_time), "exit_price": tp_price}
        if hit_sl:
            return {"trade_outcome": "sl", "exit_time": pd.Timestamp(row.open_time), "exit_price": sl_price}
    return {"trade_outcome": None, "exit_time": None, "exit_price": None}


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
