from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

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
_INDEPENDENT_OUTCOME_COLUMNS: tuple[str, ...] = ("signal_id", *_EXECUTION_OUTPUT_COLUMNS)


@dataclass(frozen=True)
class _SymbolMarketBars:
    open_time: np.ndarray
    open_time_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    time_to_pos: dict[pd.Timestamp, int]
    minute_start_by_15m_pos: np.ndarray | None = None
    minute_end_by_15m_pos: np.ndarray | None = None


@dataclass(frozen=True)
class ExecutionMarketView:
    bars_15m_by_symbol: dict[str, _SymbolMarketBars]
    bars_1m_by_symbol: dict[str, _SymbolMarketBars]


class OneSecondBarsFetcher(Protocol):
    def fetch(self, symbol: str, minute_start: pd.Timestamp) -> pd.DataFrame: ...


def build_execution_market_view(
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
) -> ExecutionMarketView:
    bars_15m = prepare_intraday_bars_frame(bars_15m_df, "15m")
    bars_1m = prepare_intraday_bars_frame(bars_1m_df, "1m")
    bars_1m_by_symbol: dict[str, _SymbolMarketBars] = {}
    for symbol, grp in bars_1m.groupby("symbol", sort=False):
        times = grp["open_time"].to_numpy(copy=True)
        times_ns = grp["open_time"].astype("int64").to_numpy(copy=True)
        bars_1m_by_symbol[str(symbol)] = _SymbolMarketBars(
            open_time=times,
            open_time_ns=times_ns,
            open=grp["open"].to_numpy(dtype=float, copy=True),
            high=grp["high"].to_numpy(dtype=float, copy=True),
            low=grp["low"].to_numpy(dtype=float, copy=True),
            close=grp["close"].to_numpy(dtype=float, copy=True),
            time_to_pos={
                pd.Timestamp(ts): int(i) for i, ts in enumerate(times.tolist())
            },
            minute_start_by_15m_pos=None,
            minute_end_by_15m_pos=None,
        )
    bars_15m_by_symbol: dict[str, _SymbolMarketBars] = {}
    fifteen_min_ns = int(pd.Timedelta(minutes=15).value)
    for symbol, grp in bars_15m.groupby("symbol", sort=False):
        symbol_str = str(symbol)
        times = grp["open_time"].to_numpy(copy=True)
        times_ns = grp["open_time"].astype("int64").to_numpy(copy=True)
        minute_view = bars_1m_by_symbol.get(symbol_str)
        minute_start: np.ndarray | None
        minute_end: np.ndarray | None
        if minute_view is None:
            minute_start = None
            minute_end = None
        else:
            minute_start = np.searchsorted(
                minute_view.open_time_ns, times_ns, side="left"
            ).astype(np.int64, copy=False)
            minute_end = np.searchsorted(
                minute_view.open_time_ns, times_ns + fifteen_min_ns, side="left"
            ).astype(np.int64, copy=False)
        bars_15m_by_symbol[symbol_str] = _SymbolMarketBars(
            open_time=times,
            open_time_ns=times_ns,
            open=grp["open"].to_numpy(dtype=float, copy=True),
            high=grp["high"].to_numpy(dtype=float, copy=True),
            low=grp["low"].to_numpy(dtype=float, copy=True),
            close=grp["close"].to_numpy(dtype=float, copy=True),
            time_to_pos={
                pd.Timestamp(ts): int(i) for i, ts in enumerate(times.tolist())
            },
            minute_start_by_15m_pos=minute_start,
            minute_end_by_15m_pos=minute_end,
        )
    return ExecutionMarketView(
        bars_15m_by_symbol=bars_15m_by_symbol, bars_1m_by_symbol=bars_1m_by_symbol
    )


def replay_independent_short_signals(
    decision_df: pd.DataFrame,
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: OneSecondBarsFetcher | None = None,
    market_view: ExecutionMarketView | None = None,
) -> pd.DataFrame:
    _require_columns(
        decision_df,
        (
            "signal_id",
            "episode_id",
            "symbol",
            "context_bar_open_time",
            "decision_time",
            "entry_bar_open_time",
        ),
        "decision_df",
    )
    decisions = (
        decision_df.copy()
        .reset_index(drop=False)
        .rename(columns={"index": "_row_order"})
    )
    decisions["context_bar_open_time"] = pd.to_datetime(
        decisions["context_bar_open_time"], utc=True, errors="raise"
    )
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    decisions["entry_bar_open_time"] = pd.to_datetime(
        decisions["entry_bar_open_time"], utc=True, errors="raise"
    )
    resolved_market_view = market_view or build_execution_market_view(
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
    )
    for column in _EXECUTION_OUTPUT_COLUMNS:
        decisions[column] = pd.NA
    execution_status = np.full(len(decisions), "missing_entry_bar", dtype=object)
    entry_price_out = np.full(len(decisions), np.nan, dtype=float)
    exit_time_out = np.array([pd.NaT] * len(decisions), dtype=object)
    exit_price_out = np.full(len(decisions), np.nan, dtype=float)
    trade_outcome_out = np.array([pd.NA] * len(decisions), dtype=object)
    trade_pnl_pct_out = np.full(len(decisions), np.nan, dtype=float)
    mfe_pct_out = np.full(len(decisions), np.nan, dtype=float)
    mae_pct_out = np.full(len(decisions), np.nan, dtype=float)
    holding_bars_out = np.array([pd.NA] * len(decisions), dtype=object)
    hold_window_ns = int(pd.Timedelta(minutes=int(execution_contract.max_hold_bars) * 15).value)
    ordered = decisions.sort_values(
        ["entry_bar_open_time", "signal_id"], kind="mergesort"
    )
    for row in ordered.itertuples(index=True):
        idx = int(row.Index)
        symbol = str(row.symbol)
        entry_time = pd.Timestamp(row.entry_bar_open_time)
        symbol_15m = resolved_market_view.bars_15m_by_symbol.get(symbol)
        if symbol_15m is None:
            continue
        entry_pos = symbol_15m.time_to_pos.get(entry_time)
        if entry_pos is None:
            continue
        entry_price = float(symbol_15m.open[entry_pos])
        tp_price = entry_price * (1.0 - float(execution_contract.tp_pct))
        sl_price = entry_price * (1.0 + float(execution_contract.sl_pct))
        horizon_end_ns = int(entry_time.value) + hold_window_ns
        horizon_pos = int(
            np.searchsorted(symbol_15m.open_time_ns, horizon_end_ns, side="left")
        )
        if horizon_pos <= entry_pos:
            continue
        outcome_data = _replay_single_short_path(
            symbol=symbol,
            symbol_15m=symbol_15m,
            entry_pos_15m=entry_pos,
            end_pos_15m=horizon_pos,
            symbol_1m=resolved_market_view.bars_1m_by_symbol.get(symbol),
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
        execution_status[idx] = "executed"
        entry_price_out[idx] = entry_price
        exit_time_out[idx] = exit_time
        exit_price_out[idx] = exit_price
        trade_outcome_out[idx] = trade_outcome
        trade_pnl_pct_out[idx] = trade_pnl_pct
        mfe_pct_out[idx] = float(outcome_data["mfe_pct"])
        mae_pct_out[idx] = float(outcome_data["mae_pct"])
        holding_bars_out[idx] = int(outcome_data["holding_bars"])
    decisions["execution_status"] = execution_status
    decisions["entry_price"] = entry_price_out
    decisions["exit_time"] = exit_time_out
    decisions["exit_price"] = exit_price_out
    decisions["trade_outcome"] = trade_outcome_out
    decisions["trade_pnl_pct"] = trade_pnl_pct_out
    decisions["mfe_pct"] = mfe_pct_out
    decisions["mae_pct"] = mae_pct_out
    decisions["holding_bars"] = holding_bars_out
    out = decisions.sort_values("_row_order", kind="mergesort")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    return out.loc[:, list(_INDEPENDENT_OUTCOME_COLUMNS)].reset_index(drop=True)


def replay_short_signals_with_symbol_lock_precomputed(
    decision_df: pd.DataFrame,
    independent_outcomes_df: pd.DataFrame,
    emit_summary_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(decision_df, _DECISION_REQUIRED_COLUMNS, "decision_df")
    _require_columns(
        independent_outcomes_df, _INDEPENDENT_OUTCOME_COLUMNS, "independent_outcomes_df"
    )
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
    merged = decisions.merge(
        independent_outcomes_df.loc[:, list(_INDEPENDENT_OUTCOME_COLUMNS)],
        on="signal_id",
        how="left",
        validate="one_to_one",
    )
    for column in _EXECUTION_OUTPUT_COLUMNS:
        merged[column] = merged[column]
    merged["execution_status"] = merged["execution_status"].fillna("missing_entry_bar")
    blocked_gate_mask = merged["gate_decision"] == "block"
    for column in _EXECUTION_OUTPUT_COLUMNS:
        merged.loc[blocked_gate_mask, column] = pd.NA
    merged.loc[blocked_gate_mask, "execution_status"] = "blocked_gate"
    lock_until_by_symbol: dict[str, pd.Timestamp] = {}
    kept_ordered = merged.loc[merged["gate_decision"] == "keep"].sort_values(
        ["entry_bar_open_time", "signal_id"], kind="mergesort"
    )
    execution_status_arr = merged["execution_status"].to_numpy(dtype=object, copy=True)
    symbol_arr = merged["symbol"].astype(str).to_numpy(dtype=object, copy=False)
    entry_arr = merged["entry_bar_open_time"].to_numpy(copy=False)
    exit_arr = merged["exit_time"].to_numpy(copy=False)
    output_cols = list(_EXECUTION_OUTPUT_COLUMNS)
    for row in kept_ordered.itertuples(index=True):
        idx = int(row.Index)
        if str(execution_status_arr[idx]) != "executed":
            continue
        symbol = str(symbol_arr[idx])
        entry_time = pd.Timestamp(entry_arr[idx])
        if (
            symbol in lock_until_by_symbol
            and entry_time <= lock_until_by_symbol[symbol]
        ):
            execution_status_arr[idx] = "blocked_symbol_lock"
            for column in output_cols:
                merged.iat[idx, merged.columns.get_loc(column)] = pd.NA
            continue
        exit_time = exit_arr[idx]
        if pd.notna(exit_time):
            lock_until_by_symbol[symbol] = pd.Timestamp(exit_time)
    merged["execution_status"] = execution_status_arr
    decisions_out = merged.sort_values("_row_order", kind="mergesort").drop(
        columns=["_row_order"]
    )
    decisions_out["exit_time"] = pd.to_datetime(
        decisions_out["exit_time"], utc=True, errors="coerce"
    )
    decisions_out = decisions_out[
        [
            *[
                column
                for column in _DECISION_REQUIRED_COLUMNS
                if column in decisions_out.columns
            ],
            *optional_passthrough_columns,
            *[
                column
                for column in decisions_out.columns
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
        decisions_out[decisions_out["execution_status"] == "executed"]
        .copy()
        .reset_index(drop=True)
    )
    blocked_gate = int((decisions_out["execution_status"] == "blocked_gate").sum())
    blocked_symbol_lock = int(
        (decisions_out["execution_status"] == "blocked_symbol_lock").sum()
    )
    if emit_summary_log:
        log_info(
            "EXECUTION",
            (
                "execution replay done "
                f"decisions_total={len(decisions_out)} executed_total={len(executed_signals_df)} "
                f"blocked_gate={blocked_gate} blocked_symbol_lock={blocked_symbol_lock}"
            ),
        )
    return decisions_out.reset_index(drop=True), executed_signals_df


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
    market_view: ExecutionMarketView | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_market_view = market_view or build_execution_market_view(
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
    )
    independent_outcomes_df = replay_independent_short_signals(
        decision_df=decision_df,
        bars_15m_df=bars_15m_df,
        bars_1m_df=bars_1m_df,
        execution_contract=execution_contract,
        bars_1s_fetcher=bars_1s_fetcher,
        market_view=resolved_market_view,
    )
    decisions, executed_signals_df = replay_short_signals_with_symbol_lock_precomputed(
        decision_df=decision_df,
        independent_outcomes_df=independent_outcomes_df,
        emit_summary_log=False,
    )
    if not executed_signals_df.empty:
        for row in executed_signals_df.itertuples(index=False):
            _ = ExecutedSignalRef(
                signal_id=str(row.signal_id),
                symbol=str(row.symbol),
                entry_bar_open_time=pd.Timestamp(row.entry_bar_open_time).to_pydatetime(),
                exit_time=pd.Timestamp(row.exit_time).to_pydatetime(),
                trade_outcome=TradeOutcome(str(row.trade_outcome)),
            )
    if emit_summary_log:
        blocked_gate = int((decisions["execution_status"] == "blocked_gate").sum())
        blocked_symbol_lock = int(
            (decisions["execution_status"] == "blocked_symbol_lock").sum()
        )
        log_info(
            "EXECUTION",
            (
                "execution replay done "
                f"decisions_total={len(decisions)} executed_total={len(executed_signals_df)} "
                f"blocked_gate={blocked_gate} blocked_symbol_lock={blocked_symbol_lock}"
            ),
        )
    return decisions, executed_signals_df


def _replay_single_short_path(
    symbol: str,
    symbol_15m: _SymbolMarketBars,
    entry_pos_15m: int,
    end_pos_15m: int,
    symbol_1m: _SymbolMarketBars | None,
    bars_1s_fetcher: OneSecondBarsFetcher | None,
    tp_price: float,
    sl_price: float,
    entry_price: float,
) -> dict[str, Any]:
    trade_outcome = "timeout"
    last_pos = end_pos_15m - 1
    exit_time = pd.Timestamp(symbol_15m.open_time[last_pos])
    exit_price = float(symbol_15m.close[last_pos])
    holding_bars = int(end_pos_15m - entry_pos_15m)
    path_lows: list[float] = []
    path_highs: list[float] = []
    for pos in range(entry_pos_15m, end_pos_15m):
        bar_time = pd.Timestamp(symbol_15m.open_time[pos])
        bar_low = float(symbol_15m.low[pos])
        bar_high = float(symbol_15m.high[pos])
        bar_close = float(symbol_15m.close[pos])
        path_lows.append(bar_low)
        path_highs.append(bar_high)
        hit_tp = bar_low <= tp_price
        hit_sl = bar_high >= sl_price
        if hit_tp and hit_sl:
            resolution = _resolve_intra_15m_touch(
                symbol=symbol,
                symbol_15m=symbol_15m,
                bar_pos_15m=pos,
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
    symbol_15m: _SymbolMarketBars,
    bar_pos_15m: int,
    symbol_1m: _SymbolMarketBars | None,
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
    minute_start_by_15m_pos = symbol_15m.minute_start_by_15m_pos
    minute_end_by_15m_pos = symbol_15m.minute_end_by_15m_pos
    if minute_start_by_15m_pos is None or minute_end_by_15m_pos is None:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    minute_start_idx = int(minute_start_by_15m_pos[bar_pos_15m])
    minute_end_idx = int(minute_end_by_15m_pos[bar_pos_15m])
    if minute_end_idx <= minute_start_idx:
        return {
            "trade_outcome": None,
            "exit_time": None,
            "exit_price": None,
            "path_lows": [],
            "path_highs": [],
        }
    lows: list[float] = []
    highs: list[float] = []
    for minute_pos in range(minute_start_idx, minute_end_idx):
        minute_start = pd.Timestamp(symbol_1m.open_time[minute_pos])
        minute_low = float(symbol_1m.low[minute_pos])
        minute_high = float(symbol_1m.high[minute_pos])
        minute_close = float(symbol_1m.close[minute_pos])
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
