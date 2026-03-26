from typing import Any, Protocol

import pandas as pd

from pump_end_v2.contracts import ExecutedSignalRef, ExecutionContract, TradeOutcome
from pump_end_v2.logging import log_info

_BARS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
)
_DECISION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "gate_decision",
)
_DECISION_OPTIONAL_PASSTHROUGH_COLUMNS: tuple[str, ...] = (
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


class OneSecondBarsFetcher(Protocol):
    def fetch(self, symbol: str, minute_start: pd.Timestamp) -> pd.DataFrame: ...


def prepare_intraday_bars_frame(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    if timeframe_label not in {"1m", "1s", "15m"}:
        raise ValueError("timeframe_label must be one of: 15m, 1m, 1s")
    _require_columns(df, _BARS_REQUIRED_COLUMNS, f"bars_{timeframe_label}_df")
    frame = df.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.sort_values(["symbol", "open_time"], kind="mergesort").reset_index(
        drop=True
    )
    dup_mask = frame.duplicated(subset=["symbol", "open_time"], keep=False)
    if dup_mask.any():
        sample = frame.loc[dup_mask, ["symbol", "open_time"]].head(3).to_dict("records")
        raise ValueError(
            f"duplicate open_time within symbol for {timeframe_label}: sample={sample}"
        )
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
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: OneSecondBarsFetcher | None = None,
    emit_summary_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(decision_df, _DECISION_REQUIRED_COLUMNS, "decision_df")
    decisions = (
        decision_df.copy()
        .reset_index(drop=False)
        .rename(columns={"index": "_row_order"})
    )
    optional_passthrough_columns = [
        column
        for column in _DECISION_OPTIONAL_PASSTHROUGH_COLUMNS
        if column in decisions.columns
    ]
    decisions["context_bar_open_time"] = pd.to_datetime(
        decisions["context_bar_open_time"], utc=True, errors="raise"
    )
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    decisions["entry_bar_open_time"] = pd.to_datetime(
        decisions["entry_bar_open_time"], utc=True, errors="raise"
    )
    decisions["gate_decision"] = decisions["gate_decision"].astype(str).str.lower()
    bars_15m = prepare_intraday_bars_frame(bars_15m_df, "15m")
    bars_1m = prepare_intraday_bars_frame(bars_1m_df, "1m")
    for column in _EXECUTION_OUTPUT_COLUMNS:
        decisions[column] = pd.NA
    decisions["execution_status"] = "pending"
    blocked_gate_mask = decisions["gate_decision"] == "block"
    decisions.loc[blocked_gate_mask, "execution_status"] = "blocked_gate"
    decisions.loc[blocked_gate_mask, "trade_outcome"] = pd.NA
    lock_until_by_symbol: dict[str, pd.Timestamp] = {}
    kept_indices = decisions.index[decisions["gate_decision"] == "keep"].tolist()
    kept_ordered = decisions.loc[kept_indices].sort_values(
        ["entry_bar_open_time", "signal_id"], kind="mergesort"
    )
    bars_15m_by_symbol = {
        symbol: grp.reset_index(drop=True)
        for symbol, grp in bars_15m.groupby("symbol", sort=False)
    }
    bars_1m_by_symbol = {
        symbol: grp.reset_index(drop=True)
        for symbol, grp in bars_1m.groupby("symbol", sort=False)
    }
    hold_window_delta = pd.Timedelta(minutes=int(execution_contract.max_hold_bars) * 15)
    for idx, row in kept_ordered.iterrows():
        symbol = str(row["symbol"])
        entry_time = pd.Timestamp(row["entry_bar_open_time"])
        if (
            symbol in lock_until_by_symbol
            and entry_time <= lock_until_by_symbol[symbol]
        ):
            decisions.at[idx, "execution_status"] = "blocked_symbol_lock"
            continue
        symbol_15m = bars_15m_by_symbol.get(symbol)
        if symbol_15m is None:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        entry_match = symbol_15m[symbol_15m["open_time"] == entry_time]
        if entry_match.empty:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        entry_bar = entry_match.iloc[0]
        entry_price = float(entry_bar["open"])
        tp_price = entry_price * (1.0 - float(execution_contract.tp_pct))
        sl_price = entry_price * (1.0 + float(execution_contract.sl_pct))
        horizon_end = entry_time + hold_window_delta
        path_bars = symbol_15m[
            (symbol_15m["open_time"] >= entry_time)
            & (symbol_15m["open_time"] < horizon_end)
        ].copy()
        if path_bars.empty:
            decisions.at[idx, "execution_status"] = "missing_entry_bar"
            continue
        outcome_data = _replay_single_short_path(
            symbol=symbol,
            path_15m=path_bars,
            symbol_1m=bars_1m_by_symbol.get(symbol),
            bars_1s_fetcher=bars_1s_fetcher,
            tp_price=tp_price,
            sl_price=sl_price,
            entry_price=entry_price,
        )
        trade_outcome = str(outcome_data["trade_outcome"])
        exit_time = outcome_data["exit_time"]
        if trade_outcome == "ambiguous":
            exit_price = float(entry_price)
            trade_pnl_pct = 0.0
        else:
            exit_price = float(outcome_data["exit_price"])
            trade_pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
        decisions.at[idx, "execution_status"] = "executed"
        decisions.at[idx, "entry_price"] = entry_price
        decisions.at[idx, "exit_time"] = exit_time
        decisions.at[idx, "exit_price"] = exit_price
        decisions.at[idx, "trade_outcome"] = trade_outcome
        decisions.at[idx, "trade_pnl_pct"] = trade_pnl_pct
        decisions.at[idx, "mfe_pct"] = outcome_data["mfe_pct"]
        decisions.at[idx, "mae_pct"] = outcome_data["mae_pct"]
        decisions.at[idx, "holding_bars"] = int(outcome_data["holding_bars"])
        _ = ExecutedSignalRef(
            signal_id=str(row["signal_id"]),
            symbol=symbol,
            entry_bar_open_time=entry_time.to_pydatetime(),
            exit_time=pd.Timestamp(exit_time).to_pydatetime(),
            trade_outcome=TradeOutcome(trade_outcome),
        )
        lock_until_by_symbol[symbol] = pd.Timestamp(exit_time)
    decisions = decisions.sort_values("_row_order", kind="mergesort").drop(
        columns=["_row_order"]
    )
    decisions = decisions[
        [
            *[
                column
                for column in _DECISION_REQUIRED_COLUMNS
                if column in decisions.columns
            ],
            *optional_passthrough_columns,
            *[
                column
                for column in decisions.columns
                if column
                not in {
                    *_DECISION_REQUIRED_COLUMNS,
                    *optional_passthrough_columns,
                    *_EXECUTION_OUTPUT_COLUMNS,
                }
            ],
            *_EXECUTION_OUTPUT_COLUMNS,
        ]
    ]
    executed_signals_df = (
        decisions[decisions["execution_status"] == "executed"]
        .copy()
        .reset_index(drop=True)
    )
    blocked_gate = int((decisions["execution_status"] == "blocked_gate").sum())
    blocked_symbol_lock = int(
        (decisions["execution_status"] == "blocked_symbol_lock").sum()
    )
    if emit_summary_log:
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
    path_15m: pd.DataFrame,
    symbol_1m: pd.DataFrame | None,
    bars_1s_fetcher: OneSecondBarsFetcher | None,
    tp_price: float,
    sl_price: float,
    entry_price: float,
) -> dict[str, Any]:
    trade_outcome = "timeout"
    exit_time = pd.Timestamp(path_15m["open_time"].iloc[-1])
    exit_price = float(path_15m["close"].iloc[-1])
    holding_bars = int(len(path_15m))
    path_lows: list[float] = []
    path_highs: list[float] = []
    for bar in path_15m.itertuples(index=False):
        bar_time = pd.Timestamp(bar.open_time)
        bar_low = float(bar.low)
        bar_high = float(bar.high)
        bar_close = float(bar.close)
        path_lows.append(bar_low)
        path_highs.append(bar_high)
        hit_tp = bar_low <= tp_price
        hit_sl = bar_high >= sl_price
        if hit_tp and hit_sl:
            resolution = _resolve_intra_15m_touch(
                symbol=symbol,
                bar_15m_start=bar_time,
                symbol_1m=symbol_1m,
                bars_1s_fetcher=bars_1s_fetcher,
                tp_price=tp_price,
                sl_price=sl_price,
            )
            if resolution["path_lows"]:
                path_lows[-1] = min(resolution["path_lows"])
                path_highs[-1] = max(resolution["path_highs"])
            if resolution["trade_outcome"] is None:
                trade_outcome = "ambiguous"
                exit_time = bar_time
                exit_price = bar_close
            else:
                trade_outcome = str(resolution["trade_outcome"])
                exit_time = pd.Timestamp(resolution["exit_time"])
                exit_price = float(resolution["exit_price"])
            holding_bars = len(path_lows)
            break
        if hit_tp:
            trade_outcome = "tp"
            exit_time = bar_time
            exit_price = tp_price
            holding_bars = len(path_lows)
            break
        if hit_sl:
            trade_outcome = "sl"
            exit_time = bar_time
            exit_price = sl_price
            holding_bars = len(path_lows)
            break
    mfe_pct = 0.0
    mae_pct = 0.0
    if path_lows and path_highs:
        mfe_pct = float(((entry_price - min(path_lows)) / entry_price) * 100.0)
        mae_pct = float(((max(path_highs) - entry_price) / entry_price) * 100.0)
    return {
        "symbol": symbol,
        "trade_outcome": trade_outcome,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "holding_bars": holding_bars,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }


def _resolve_intra_15m_touch(
    symbol: str,
    bar_15m_start: pd.Timestamp,
    symbol_1m: pd.DataFrame | None,
    bars_1s_fetcher: OneSecondBarsFetcher | None,
    tp_price: float,
    sl_price: float,
) -> dict[str, Any]:
    if symbol_1m is None:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    bar_15m_end = bar_15m_start + pd.Timedelta(minutes=15)
    minute_bars = symbol_1m[
        (symbol_1m["open_time"] >= bar_15m_start) & (symbol_1m["open_time"] < bar_15m_end)
    ]
    if minute_bars.empty:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    lows: list[float] = []
    highs: list[float] = []
    for row in minute_bars.itertuples(index=False):
        minute_start = pd.Timestamp(row.open_time)
        minute_low = float(row.low)
        minute_high = float(row.high)
        minute_close = float(row.close)
        lows.append(minute_low)
        highs.append(minute_high)
        hit_tp = minute_low <= tp_price
        hit_sl = minute_high >= sl_price
        if hit_tp and hit_sl:
            second_outcome = _resolve_intraminute_touch(
                symbol=symbol,
                minute_start=minute_start,
                bars_1s_fetcher=bars_1s_fetcher,
                tp_price=tp_price,
                sl_price=sl_price,
            )
            if second_outcome["path_lows"]:
                lows[-1] = min(second_outcome["path_lows"])
                highs[-1] = max(second_outcome["path_highs"])
            if second_outcome["trade_outcome"] is None:
                return {
                    "trade_outcome": "ambiguous",
                    "exit_time": minute_start,
                    "exit_price": minute_close,
                    "path_lows": lows,
                    "path_highs": highs,
                }
            return {
                "trade_outcome": second_outcome["trade_outcome"],
                "exit_time": second_outcome["exit_time"],
                "exit_price": second_outcome["exit_price"],
                "path_lows": lows,
                "path_highs": highs,
            }
        if hit_tp:
            return {
                "trade_outcome": "tp",
                "exit_time": minute_start,
                "exit_price": tp_price,
                "path_lows": lows,
                "path_highs": highs,
            }
        if hit_sl:
            return {
                "trade_outcome": "sl",
                "exit_time": minute_start,
                "exit_price": sl_price,
                "path_lows": lows,
                "path_highs": highs,
            }
    return {
        "trade_outcome": None,
        "exit_time": None,
        "exit_price": None,
        "path_lows": lows,
        "path_highs": highs,
    }


def _resolve_intraminute_touch(
    symbol: str,
    minute_start: pd.Timestamp,
    bars_1s_fetcher: OneSecondBarsFetcher | None,
    tp_price: float,
    sl_price: float,
) -> dict[str, Any]:
    if bars_1s_fetcher is None:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    minute_end = minute_start + pd.Timedelta(minutes=1)
    minute_bars = bars_1s_fetcher.fetch(symbol=symbol, minute_start=minute_start)
    minute_bars = minute_bars[
        (minute_bars["open_time"] >= minute_start) & (minute_bars["open_time"] < minute_end)
    ]
    if minute_bars.empty:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    lows: list[float] = []
    highs: list[float] = []
    for row in minute_bars.itertuples(index=False):
        sec_low = float(row.low)
        sec_high = float(row.high)
        lows.append(sec_low)
        highs.append(sec_high)
        hit_tp = sec_low <= tp_price
        hit_sl = sec_high >= sl_price
        if hit_tp and hit_sl:
            return {
                "trade_outcome": "ambiguous",
                "exit_time": pd.Timestamp(row.open_time),
                "exit_price": float(row.close),
                "path_lows": lows,
                "path_highs": highs,
            }
        if hit_tp:
            return {
                "trade_outcome": "tp",
                "exit_time": pd.Timestamp(row.open_time),
                "exit_price": tp_price,
                "path_lows": lows,
                "path_highs": highs,
            }
        if hit_sl:
            return {
                "trade_outcome": "sl",
                "exit_time": pd.Timestamp(row.open_time),
                "exit_price": sl_price,
                "path_lows": lows,
                "path_highs": highs,
            }
    return {
        "trade_outcome": None,
        "exit_time": None,
        "exit_price": None,
        "path_lows": lows,
        "path_highs": highs,
    }


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
