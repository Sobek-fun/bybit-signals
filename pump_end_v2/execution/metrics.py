import pandas as pd

from pump_end_v2.logging import log_info

_EXECUTED_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "entry_bar_open_time",
    "trade_outcome",
    "trade_pnl_pct",
)


def build_execution_metrics(
    executed_signals_df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> dict[str, float]:
    _require_columns(
        executed_signals_df, _EXECUTED_REQUIRED_COLUMNS, "executed_signals_df"
    )
    frame = _prepare_frame(executed_signals_df)
    signals = int(len(frame))
    symbols = int(frame["symbol"].nunique()) if signals > 0 else 0
    tp = int((frame["trade_outcome"] == "tp").sum())
    sl = int((frame["trade_outcome"] == "sl").sum())
    timeout = int((frame["trade_outcome"] == "timeout").sum())
    ambiguous = int((frame["trade_outcome"] == "ambiguous").sum())
    resolved = tp + sl
    tp_rate_resolved = _safe_ratio(tp, resolved)
    sl_rate_resolved = _safe_ratio(sl, resolved)
    pnl_sum = float(frame["trade_pnl_pct"].sum()) if signals > 0 else 0.0
    expectancy_all = (
        float(frame["trade_pnl_pct"].mean()) if signals > 0 else float("nan")
    )
    resolved_mask = frame["trade_outcome"].isin(["tp", "sl"])
    expectancy_resolved = (
        float(frame.loc[resolved_mask, "trade_pnl_pct"].mean())
        if resolved > 0
        else float("nan")
    )
    pos_sum = (
        float(frame.loc[frame["trade_pnl_pct"] > 0, "trade_pnl_pct"].sum())
        if signals > 0
        else 0.0
    )
    neg_sum_abs = (
        float(abs(frame.loc[frame["trade_pnl_pct"] < 0, "trade_pnl_pct"].sum()))
        if signals > 0
        else 0.0
    )
    profit_factor = (
        float(pos_sum / neg_sum_abs) if (pos_sum > 0.0 and neg_sum_abs > 0.0) else 0.0
    )
    max_losing_streak = float(_compute_max_losing_streak(frame))
    eval_window_days = _resolve_eval_window_days(
        window_start=window_start, window_end=window_end, window_days=window_days
    )
    signals_per_30d = float(_compute_signals_per_30d(len(frame), eval_window_days))
    report_6h = build_execution_window_report(frame, 6)
    report_24h = build_execution_window_report(frame, 24)
    worst_6h_pnl = (
        float(report_6h["pnl_sum"].min()) if not report_6h.empty else float("nan")
    )
    worst_24h_pnl = (
        float(report_24h["pnl_sum"].min()) if not report_24h.empty else float("nan")
    )
    metrics = {
        "signals": float(signals),
        "symbols": float(symbols),
        "signals_per_30d": signals_per_30d,
        "tp": float(tp),
        "sl": float(sl),
        "timeout": float(timeout),
        "ambiguous": float(ambiguous),
        "tp_rate_resolved": float(tp_rate_resolved),
        "sl_rate_resolved": float(sl_rate_resolved),
        "max_losing_streak": max_losing_streak,
        "pnl_sum": pnl_sum,
        "expectancy_all": float(expectancy_all),
        "expectancy_resolved": float(expectancy_resolved),
        "profit_factor": profit_factor,
        "worst_6h_pnl": worst_6h_pnl,
        "worst_24h_pnl": worst_24h_pnl,
    }
    log_info(
        "EXECUTION",
        (
            "execution metrics done "
            f"signals={signals} signals_per_30d={metrics['signals_per_30d']:.6f} pnl_sum={pnl_sum:.6f}"
        ),
    )
    return metrics


def build_execution_window_report(
    executed_signals_df: pd.DataFrame, window_hours: int
) -> pd.DataFrame:
    _require_columns(
        executed_signals_df, _EXECUTED_REQUIRED_COLUMNS, "executed_signals_df"
    )
    frame = _prepare_frame(executed_signals_df)
    if frame.empty:
        return pd.DataFrame(columns=["window_end_time", "trades", "pnl_sum"])
    window = pd.Timedelta(hours=int(window_hours))
    rows: list[dict[str, object]] = []
    times = frame["entry_bar_open_time"].tolist()
    pnl = frame["trade_pnl_pct"].tolist()
    for idx, current_time in enumerate(times):
        left = 0
        window_start = current_time - window
        while left <= idx and times[left] < window_start:
            left += 1
        pnl_sum = float(sum(pnl[left : idx + 1]))
        rows.append(
            {
                "window_end_time": current_time,
                "trades": int(idx - left + 1),
                "pnl_sum": pnl_sum,
            }
        )
    out = (
        pd.DataFrame(rows)
        .sort_values("window_end_time", kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def build_execution_symbol_report(executed_signals_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        executed_signals_df, _EXECUTED_REQUIRED_COLUMNS, "executed_signals_df"
    )
    frame = _prepare_frame(executed_signals_df)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signals",
                "tp",
                "sl",
                "timeout",
                "ambiguous",
                "pnl_sum",
                "expectancy_all",
            ]
        )
    groups = frame.groupby("symbol", sort=True, dropna=False)
    rows: list[dict[str, float | str]] = []
    for symbol, grp in groups:
        rows.append(
            {
                "symbol": str(symbol),
                "signals": float(len(grp)),
                "tp": float((grp["trade_outcome"] == "tp").sum()),
                "sl": float((grp["trade_outcome"] == "sl").sum()),
                "timeout": float((grp["trade_outcome"] == "timeout").sum()),
                "ambiguous": float((grp["trade_outcome"] == "ambiguous").sum()),
                "pnl_sum": float(grp["trade_pnl_pct"].sum()),
                "expectancy_all": (
                    float(grp["trade_pnl_pct"].mean()) if len(grp) > 0 else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("symbol", kind="mergesort")
        .reset_index(drop=True)
    )


def build_execution_monthly_report(executed_signals_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        executed_signals_df, _EXECUTED_REQUIRED_COLUMNS, "executed_signals_df"
    )
    columns = [
        "month",
        "signals",
        "tp",
        "sl",
        "timeout",
        "ambiguous",
        "pnl_sum",
        "expectancy_all",
    ]
    frame = _prepare_frame(executed_signals_df)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    month_series = frame["entry_bar_open_time"].dt.strftime("%Y-%m")
    grouped = frame.assign(month=month_series).groupby("month", sort=True, dropna=False)
    rows: list[dict[str, float | str]] = []
    for month, grp in grouped:
        rows.append(
            {
                "month": str(month),
                "signals": float(len(grp)),
                "tp": float((grp["trade_outcome"] == "tp").sum()),
                "sl": float((grp["trade_outcome"] == "sl").sum()),
                "timeout": float((grp["trade_outcome"] == "timeout").sum()),
                "ambiguous": float((grp["trade_outcome"] == "ambiguous").sum()),
                "pnl_sum": float(grp["trade_pnl_pct"].sum()),
                "expectancy_all": (
                    float(grp["trade_pnl_pct"].mean()) if len(grp) > 0 else float("nan")
                ),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("month", kind="mergesort").reset_index(drop=True)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return frame
    frame["entry_bar_open_time"] = pd.to_datetime(
        frame["entry_bar_open_time"], utc=True, errors="coerce"
    )
    frame["trade_pnl_pct"] = pd.to_numeric(frame["trade_pnl_pct"], errors="coerce")
    frame["trade_outcome"] = frame["trade_outcome"].astype(str).str.lower()
    frame = frame.dropna(subset=["entry_bar_open_time", "trade_pnl_pct"]).sort_values(
        "entry_bar_open_time", kind="mergesort"
    )
    return frame.reset_index(drop=True)


def _compute_signals_per_30d(signals_count: int, eval_window_days: float) -> float:
    if signals_count <= 0:
        return 0.0
    safe_window_days = max(float(eval_window_days), 1e-9)
    return float(float(signals_count) * 30.0 / safe_window_days)


def _resolve_eval_window_days(
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
    window_days: float | None,
) -> float:
    if window_days is not None:
        resolved = float(window_days)
        if resolved <= 0.0:
            raise ValueError("window_days must be positive")
        return resolved
    if window_start is None or window_end is None:
        return 1.0
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    if end <= start:
        raise ValueError("window_end must be greater than window_start")
    return float((end - start) / pd.Timedelta(days=1))


def _compute_max_losing_streak(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    streak = 0
    best = 0
    for row in frame.itertuples(index=False):
        pnl = float(row.trade_pnl_pct)
        if pnl < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
