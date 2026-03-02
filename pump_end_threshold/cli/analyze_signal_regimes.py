#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from pump_end_threshold.features.feature_builder import PumpFeatureBuilder
from pump_end_threshold.features.params import DEFAULT_PUMP_PARAMS
from pump_end_threshold.infra.clickhouse import DataLoader


# ======================================================================================
# HOW TO USE
# 1) Drop one or more signal files into ./analysis_inputs/
#    Supported: .csv, .parquet
#    Required columns:
#      - symbol
#      - one of: timestamp / open_time / event_open_time / signal_time
# 2) Run:
#      python -m pump_end_threshold.cli.analyze_signal_regimes --ch-db http://user:pass@host:8123/db
#
# The script is intentionally opinionated and keeps only one CLI parameter (--ch-db).
# All other knobs live below as constants so they are easy to edit if your execution
# semantics differ from the defaults.
# ======================================================================================

INPUT_PATTERNS = [
    "analysis_inputs/**/*.csv",
    "analysis_inputs/**/*.parquet",
    "signals/**/*.csv",
    "signals/**/*.parquet",
    "*signals*.csv",
    "*signals*.parquet",
    "*signal*.csv",
    "*signal*.parquet",
]

OUTPUT_DIR = Path("analysis_outputs/pump_end_regime")

# Trading assumptions for analytics.
# Default entry is NEXT bar open. This is conservative and avoids using the same bar
# as the signal timestamp if the real execution time is bar-close / delayed confirmation.
# If your prod entry is "same bar open", change ENTRY_BAR_SHIFT to 0.
ENTRY_BAR_SHIFT = 1
TP_PCT = 0.045
SL_PCT = 0.10
MAX_HOLD_BARS = 96  # 24h on 15m bars
STREAK_BREAK_GAP_HOURS = 12

# Feature extraction defaults for "what the model saw" + extra analytics context.
WINDOW_BARS = 30
WARMUP_BARS = 150
FEATURE_SET = "extended"
MARKET_SYMBOL = "BTCUSDT"
REFERENCE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SIGNAL_CANDLES_BATCH_SIZE = 40
BREADTH_BATCH_SIZE = 35

# Market/breadth lookbacks expressed in 15m bars.
RET_WINDOWS = (1, 4, 16, 96)
VOL_MEDIAN_WINDOW = 20
RANGE_LOOKBACK = 16
MA_LOOKBACK = 32
HIGH_LOOKBACK = 96
LOW_LOOKBACK = 96
GREEN_BIG_THRESHOLD_1 = 0.02
BREADTH_THRESHOLD_RET4 = 0.02
BREADTH_THRESHOLD_RET16 = 0.05
BREADTH_THRESHOLD_RET96 = 0.10
BREADTH_VOL_SPIKE = 3.0
NEAR_HIGH_EPS = 0.01

# Candidate future-regime labels for the later gate model.
NEXT_SIGNAL_WINDOWS = (3, 5)
NEXT_TIME_WINDOWS_HOURS = (6, 12)
MIN_RESOLVED_SIGNALS_FOR_TIME_TARGET = {6: 3, 12: 4}

# Analysis model / rankings.
MAX_MODEL_FEATURES = 150
MIN_FEATURE_ROWS = 40

# Focus windows described by the user.
FOCUS_MONTH_DAY_WINDOWS = [
    ("focus_feb12", "02-12 02:00:00", "02-12 23:45:00"),
    ("focus_feb24_25", "02-24 01:30:00", "02-25 10:45:00"),
]


# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------
def log(message: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def to_utc_naive(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_localize(None)


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        return s
    if not s.endswith("USDT") and "/" not in s and "-" not in s:
        s = f"{s}USDT"
    return s


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def positive_streak(values: pd.Series) -> pd.Series:
    arr = values.fillna(False).to_numpy(dtype=bool)
    streak = np.zeros(len(arr), dtype=np.int32)
    for i in range(len(arr)):
        if arr[i]:
            streak[i] = 1 if i == 0 else streak[i - 1] + 1
        else:
            streak[i] = 0
    return pd.Series(streak, index=values.index)


def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def safe_div(num, den):
    den = np.where(np.asarray(den) == 0, np.nan, den)
    return np.asarray(num) / den


def clean_constant_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    keep = []
    for col in feature_cols:
        s = df[col]
        if s.notna().sum() < MIN_FEATURE_ROWS:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return keep


def format_float(x) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax >= 100:
        return f"{x:,.1f}"
    if ax >= 10:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.3f}"
    return f"{x:,.4f}"


def markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_empty_"
    shown = df.head(max_rows).copy()
    cols = list(shown.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in shown.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating, int, np.integer)) and not isinstance(val, bool):
                vals.append(format_float(val))
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def safe_subset(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    existing = [c for c in wanted if c in df.columns]
    if not existing:
        return pd.DataFrame()
    return df[existing].copy()


# --------------------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------------------
def discover_signal_files() -> list[Path]:
    root = Path.cwd()
    found: list[Path] = []
    seen = set()
    for pattern in INPUT_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if OUTPUT_DIR in path.parents:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return sorted(found)


def read_signal_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def standardize_signal_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    raw = df.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    time_col = None
    for candidate in ("open_time", "timestamp", "event_open_time", "signal_time", "datetime", "time"):
        if candidate in raw.columns:
            time_col = candidate
            break
    if time_col is None or "symbol" not in raw.columns:
        raise ValueError(
            f"{source}: expected columns symbol + one of open_time/timestamp/event_open_time/signal_time"
        )

    result = raw.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["open_time"] = to_utc_naive(result[time_col]).dt.floor("15min")
    result["source_file"] = source

    # Preserve useful non-conflicting columns from the source file if they exist.
    keep = ["symbol", "open_time", "source_file"]
    for extra in ("side", "strategy", "tag", "score", "model", "exchange"):
        if extra in result.columns:
            keep.append(extra)

    result = result[keep].dropna(subset=["symbol", "open_time"])
    result = result.drop_duplicates(subset=["symbol", "open_time"])
    return result


def load_signals() -> pd.DataFrame:
    files = discover_signal_files()
    if not files:
        raise FileNotFoundError(
            "No signal files found. Put CSV/Parquet files into ./analysis_inputs/ (or ./signals/)."
        )

    frames = []
    for path in files:
        log(f"loading signals: {path}")
        df = read_signal_file(path)
        frames.append(standardize_signal_frame(df, path.name))

    signals = pd.concat(frames, ignore_index=True)
    signals = signals.drop_duplicates(subset=["symbol", "open_time"]).sort_values(["open_time", "symbol"])
    signals = signals.reset_index(drop=True)
    signals["signal_id"] = signals["symbol"] + "|" + signals["open_time"].dt.strftime("%Y%m%d_%H%M%S")
    return signals


# --------------------------------------------------------------------------------------
# ClickHouse loading
# --------------------------------------------------------------------------------------
def list_usdt_symbols(loader: DataLoader, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[str]:
    query = """
    SELECT DISTINCT symbol
    FROM bybit.candles
    WHERE interval = 1
      AND open_time >= %(start)s
      AND open_time < %(end)s
      AND endsWith(symbol, 'USDT')
    ORDER BY symbol
    """
    result = loader.client.query(query, parameters={"start": start_dt.to_pydatetime(), "end": end_dt.to_pydatetime()})
    return [row[0] for row in result.result_rows]


def load_candles_dict(
    loader: DataLoader,
    symbols: list[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    batch_size: int,
    label: str,
) -> dict[str, pd.DataFrame]:
    symbols = sorted(set(symbols))
    result: dict[str, pd.DataFrame] = {}
    if not symbols:
        return result

    total_batches = math.ceil(len(symbols) / batch_size)
    for idx, batch in enumerate(chunks(symbols, batch_size), start=1):
        log(f"loading {label}: batch {idx}/{total_batches} ({len(batch)} symbols)")
        batch_map = loader.load_candles_batch(batch, start_dt.to_pydatetime(), end_dt.to_pydatetime())
        for symbol, df in batch_map.items():
            if df is None or df.empty:
                continue
            result[symbol] = df.sort_index()
    return result


# --------------------------------------------------------------------------------------
# Candle-derived context
# --------------------------------------------------------------------------------------
def compute_context_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    out[f"{prefix}ret_1"] = close.pct_change(1)
    out[f"{prefix}ret_4"] = close.pct_change(4)
    out[f"{prefix}ret_16"] = close.pct_change(16)
    out[f"{prefix}ret_96"] = close.pct_change(96)

    rolling_high_16 = close.rolling(16).max()
    rolling_high_96 = close.rolling(HIGH_LOOKBACK).max()
    rolling_low_96 = close.rolling(LOW_LOOKBACK).min()

    out[f"{prefix}drawdown_16"] = safe_div(close.to_numpy(), rolling_high_16.to_numpy()) - 1
    out[f"{prefix}drawdown_96"] = safe_div(close.to_numpy(), rolling_high_96.to_numpy()) - 1
    out[f"{prefix}dist_to_high_96"] = safe_div(close.to_numpy(), rolling_high_96.to_numpy()) - 1
    out[f"{prefix}dist_from_low_96"] = safe_div(close.to_numpy(), rolling_low_96.to_numpy()) - 1

    ret_1 = out[f"{prefix}ret_1"]
    out[f"{prefix}green_frac_16"] = ret_1.gt(0).rolling(16).mean()
    out[f"{prefix}green_frac_96"] = ret_1.gt(0).rolling(96).mean()
    out[f"{prefix}big_green_frac_16"] = ret_1.gt(GREEN_BIG_THRESHOLD_1).rolling(16).mean()
    out[f"{prefix}up_streak"] = positive_streak(ret_1.gt(0))

    range_norm = safe_div((df["high"] - df["low"]).to_numpy(), close.to_numpy())
    out[f"{prefix}range_norm"] = range_norm
    out[f"{prefix}range_mean_16"] = pd.Series(range_norm, index=df.index).rolling(RANGE_LOOKBACK).mean()

    tr = true_range(df)
    out[f"{prefix}atr_norm_14"] = safe_div(tr.rolling(14).mean().to_numpy(), close.to_numpy())

    vol_med = volume.rolling(VOL_MEDIAN_WINDOW).median()
    out[f"{prefix}vol_ratio_20"] = safe_div(volume.to_numpy(), vol_med.to_numpy())
    out[f"{prefix}vol_spike_frac_16"] = pd.Series(
        out[f"{prefix}vol_ratio_20"].to_numpy(), index=df.index
    ).gt(BREADTH_VOL_SPIKE).rolling(16).mean()

    ma_32 = close.rolling(MA_LOOKBACK).mean()
    out[f"{prefix}above_ma_32"] = safe_div(close.to_numpy(), ma_32.to_numpy()) - 1

    # Shift by 1 bar: at signal time we only want the information known before this bar opened.
    out = out.shift(1)
    return out


def merge_symbol_context(signals: pd.DataFrame, symbol_candles: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    parts = []
    preserved_cols = list(signals.columns)

    for symbol, group in signals.groupby("symbol", sort=False):
        g = group.copy()
        candles = symbol_candles.get(symbol)
        if candles is None or candles.empty:
            parts.append(g)
            continue

        context = compute_context_frame(candles, prefix=prefix).reset_index()
        if "open_time" not in context.columns:
            context = context.rename(columns={context.columns[0]: "open_time"})
        merged = g.merge(context, on="open_time", how="left")
        # If the merge added duplicate source columns, prefer original signal-side columns.
        for col in preserved_cols:
            if f"{col}_x" in merged.columns and f"{col}_y" in merged.columns:
                merged[col] = merged[f"{col}_x"]
                merged = merged.drop(columns=[f"{col}_x", f"{col}_y"])
        parts.append(merged)

    result = pd.concat(parts, ignore_index=True, sort=False)
    result = result.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    return result


def merge_reference_context(signals: pd.DataFrame, ref_candles: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if ref_candles is None or ref_candles.empty:
        return signals

    ctx = compute_context_frame(ref_candles, prefix=prefix).reset_index()
    if "open_time" not in ctx.columns:
        ctx = ctx.rename(columns={ctx.columns[0]: "open_time"})
    merged = signals.merge(ctx, on="open_time", how="left")
    return merged


# --------------------------------------------------------------------------------------
# Breadth / broad-market features
# --------------------------------------------------------------------------------------
def aggregate_breadth_batch(batch_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for _, df in batch_map.items():
        if df is None or df.empty or len(df) < HIGH_LOOKBACK:
            continue

        close = df["close"]
        volume = df["volume"]

        ret_1 = close.pct_change(1)
        ret_4 = close.pct_change(4)
        ret_16 = close.pct_change(16)
        ret_96 = close.pct_change(96)

        rolling_high_96 = close.rolling(HIGH_LOOKBACK).max()
        vol_med_20 = volume.rolling(VOL_MEDIAN_WINDOW).median()
        vol_ratio_20 = volume / vol_med_20

        batch = pd.DataFrame(index=df.index)
        batch["n_symbols"] = 1.0

        for w, ret in [(1, ret_1), (4, ret_4), (16, ret_16), (96, ret_96)]:
            batch[f"valid_ret_{w}"] = ret.notna().astype(float)
            batch[f"pos_count_{w}"] = ret.gt(0).astype(float)
            batch[f"sum_ret_{w}"] = ret.fillna(0.0)
            batch[f"sumsq_ret_{w}"] = (ret.fillna(0.0) ** 2)

        batch["count_gt_2pct_4"] = ret_4.gt(BREADTH_THRESHOLD_RET4).astype(float)
        batch["count_gt_5pct_16"] = ret_16.gt(BREADTH_THRESHOLD_RET16).astype(float)
        batch["count_gt_10pct_96"] = ret_96.gt(BREADTH_THRESHOLD_RET96).astype(float)
        batch["count_big_green_1"] = ret_1.gt(GREEN_BIG_THRESHOLD_1).astype(float)
        batch["count_near_high_96"] = (close >= rolling_high_96 * (1 - NEAR_HIGH_EPS)).astype(float)
        batch["count_vol_spike_20"] = vol_ratio_20.gt(BREADTH_VOL_SPIKE).astype(float)

        frames.append(batch.reset_index())

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    agg = full.groupby("bucket", as_index=False).sum(numeric_only=True)
    return agg


def compute_market_breadth(
    loader: DataLoader,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    symbols = list_usdt_symbols(loader, start_dt, end_dt)
    if not symbols:
        log("breadth: no USDT symbols found")
        return pd.DataFrame()

    partials = []
    total_batches = math.ceil(len(symbols) / BREADTH_BATCH_SIZE)
    for idx, batch in enumerate(chunks(symbols, BREADTH_BATCH_SIZE), start=1):
        log(f"breadth: batch {idx}/{total_batches} ({len(batch)} symbols)")
        batch_map = loader.load_candles_batch(batch, start_dt.to_pydatetime(), end_dt.to_pydatetime())
        agg = aggregate_breadth_batch(batch_map)
        if not agg.empty:
            partials.append(agg)

    if not partials:
        return pd.DataFrame()

    breadth = pd.concat(partials, ignore_index=True)
    breadth = breadth.groupby("bucket", as_index=False).sum(numeric_only=True)
    breadth = breadth.sort_values("bucket").reset_index(drop=True)

    def ratio(num_col: str, den_col: str, out_col: str) -> None:
        breadth[out_col] = breadth[num_col] / breadth[den_col].replace(0, np.nan)

    for w in (1, 4, 16, 96):
        ratio(f"pos_count_{w}", f"valid_ret_{w}", f"breadth_pos_{w}")
        ratio(f"sum_ret_{w}", f"valid_ret_{w}", f"breadth_mean_ret_{w}")
        mean = breadth[f"breadth_mean_ret_{w}"]
        var = breadth[f"sumsq_ret_{w}"] / breadth[f"valid_ret_{w}"].replace(0, np.nan) - mean**2
        breadth[f"breadth_std_ret_{w}"] = np.sqrt(np.maximum(var, 0))

    ratio("count_gt_2pct_4", "valid_ret_4", "breadth_share_gt_2pct_4")
    ratio("count_gt_5pct_16", "valid_ret_16", "breadth_share_gt_5pct_16")
    ratio("count_gt_10pct_96", "valid_ret_96", "breadth_share_gt_10pct_96")
    ratio("count_big_green_1", "valid_ret_1", "breadth_share_big_green_1")
    ratio("count_near_high_96", "n_symbols", "breadth_share_near_high_96")
    ratio("count_vol_spike_20", "n_symbols", "breadth_share_vol_spike_20")
    breadth["breadth_n_symbols"] = breadth["n_symbols"]

    keep_cols = ["bucket", "breadth_n_symbols"]
    for w in (1, 4, 16, 96):
        keep_cols.extend(
            [
                f"breadth_pos_{w}",
                f"breadth_mean_ret_{w}",
                f"breadth_std_ret_{w}",
            ]
        )
    keep_cols.extend(
        [
            "breadth_share_gt_2pct_4",
            "breadth_share_gt_5pct_16",
            "breadth_share_gt_10pct_96",
            "breadth_share_big_green_1",
            "breadth_share_near_high_96",
            "breadth_share_vol_spike_20",
        ]
    )
    breadth = breadth[keep_cols].copy()

    # Shift by 1 bar to remove lookahead.
    feature_cols = [c for c in breadth.columns if c != "bucket"]
    breadth[feature_cols] = breadth[feature_cols].shift(1)
    return breadth


# --------------------------------------------------------------------------------------
# Current model view: what the existing feature builder saw at signal time
# --------------------------------------------------------------------------------------
def build_current_model_snapshot_features(
    signals: pd.DataFrame,
    symbol_candles: dict[str, pd.DataFrame],
    btc_candles: pd.DataFrame | None,
) -> pd.DataFrame:
    builder = PumpFeatureBuilder(
        ch_dsn=None,
        window_bars=WINDOW_BARS,
        warmup_bars=WARMUP_BARS,
        feature_set=FEATURE_SET,
        params=DEFAULT_PUMP_PARAMS,
        market_symbol=MARKET_SYMBOL,
    )

    rows = []
    total_symbols = signals["symbol"].nunique()

    for i, (symbol, group) in enumerate(signals.groupby("symbol", sort=False), start=1):
        log(f"feature snapshot: {i}/{total_symbols} symbol={symbol} signals={len(group)}")
        candles = symbol_candles.get(symbol)
        if candles is None or candles.empty:
            continue

        times = sorted(group["open_time"].unique().tolist())
        try:
            symbol_rows = builder.build_many_for_inference(
                df_candles=candles,
                symbol=symbol,
                decision_open_times=times,
                btc_candles=btc_candles,
            )
        except Exception as exc:
            log(f"feature snapshot failed for {symbol}: {exc}")
            continue

        rows.extend(symbol_rows)

    if not rows:
        return pd.DataFrame(columns=["symbol", "open_time"])

    features = pd.DataFrame(rows)
    if "signal_id" not in features.columns:
        features["signal_id"] = features["symbol"] + "|" + pd.to_datetime(features["open_time"]).dt.strftime("%Y%m%d_%H%M%S")
    return features


# --------------------------------------------------------------------------------------
# Trade outcome simulation
# --------------------------------------------------------------------------------------
@dataclass
class TradeOutcome:
    entry_time: pd.Timestamp | pd.NaT
    entry_price: float | np.nan
    exit_time: pd.Timestamp | pd.NaT
    exit_price: float | np.nan
    outcome: str
    pnl_pct: float | np.nan
    pnl_pct_best: float | np.nan
    pnl_pct_worst: float | np.nan
    holding_bars: float | np.nan
    mfe_pct: float | np.nan
    mae_pct: float | np.nan


def simulate_short_trade(df: pd.DataFrame, signal_time: pd.Timestamp) -> TradeOutcome:
    if df is None or df.empty or signal_time not in df.index:
        return TradeOutcome(pd.NaT, np.nan, pd.NaT, np.nan, "no_data", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    signal_idx = df.index.get_loc(signal_time)
    if isinstance(signal_idx, slice):
        signal_idx = signal_idx.start

    entry_idx = signal_idx + ENTRY_BAR_SHIFT
    if entry_idx >= len(df):
        return TradeOutcome(pd.NaT, np.nan, pd.NaT, np.nan, "no_future", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    entry_time = pd.Timestamp(df.index[entry_idx])
    entry_price = float(df.iloc[entry_idx]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return TradeOutcome(entry_time, np.nan, pd.NaT, np.nan, "bad_entry", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    tp_price = entry_price * (1 - TP_PCT)
    sl_price = entry_price * (1 + SL_PCT)

    last_idx = min(len(df) - 1, entry_idx + MAX_HOLD_BARS - 1)
    future = df.iloc[entry_idx : last_idx + 1]

    mfe_pct = safe_div(entry_price - future["low"].min(), entry_price)
    mae_pct = safe_div(future["high"].max() - entry_price, entry_price)

    for j in range(entry_idx, last_idx + 1):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])
        current_time = pd.Timestamp(df.index[j])

        hit_tp = low <= tp_price
        hit_sl = high >= sl_price

        if hit_tp and hit_sl:
            return TradeOutcome(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=current_time,
                exit_price=np.nan,
                outcome="ambiguous",
                pnl_pct=np.nan,
                pnl_pct_best=TP_PCT,
                pnl_pct_worst=-SL_PCT,
                holding_bars=j - entry_idx + 1,
                mfe_pct=float(mfe_pct),
                mae_pct=float(mae_pct),
            )
        if hit_tp:
            return TradeOutcome(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=current_time,
                exit_price=tp_price,
                outcome="tp",
                pnl_pct=TP_PCT,
                pnl_pct_best=TP_PCT,
                pnl_pct_worst=TP_PCT,
                holding_bars=j - entry_idx + 1,
                mfe_pct=float(mfe_pct),
                mae_pct=float(mae_pct),
            )
        if hit_sl:
            return TradeOutcome(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=current_time,
                exit_price=sl_price,
                outcome="sl",
                pnl_pct=-SL_PCT,
                pnl_pct_best=-SL_PCT,
                pnl_pct_worst=-SL_PCT,
                holding_bars=j - entry_idx + 1,
                mfe_pct=float(mfe_pct),
                mae_pct=float(mae_pct),
            )

    exit_time = pd.Timestamp(df.index[last_idx])
    exit_price = float(df.iloc[last_idx]["close"])
    pnl_pct = float(safe_div(entry_price - exit_price, entry_price))
    return TradeOutcome(
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        outcome="timeout",
        pnl_pct=pnl_pct,
        pnl_pct_best=pnl_pct,
        pnl_pct_worst=pnl_pct,
        holding_bars=last_idx - entry_idx + 1,
        mfe_pct=float(mfe_pct),
        mae_pct=float(mae_pct),
    )


def compute_trade_outcomes(signals: pd.DataFrame, symbol_candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in signals.itertuples():
        outcome = simulate_short_trade(symbol_candles.get(row.symbol), row.open_time)
        rows.append(
            {
                "signal_id": row.signal_id,
                "entry_time": outcome.entry_time,
                "entry_price": outcome.entry_price,
                "exit_time": outcome.exit_time,
                "exit_price": outcome.exit_price,
                "trade_outcome": outcome.outcome,
                "trade_pnl_pct": outcome.pnl_pct,
                "trade_pnl_pct_best": outcome.pnl_pct_best,
                "trade_pnl_pct_worst": outcome.pnl_pct_worst,
                "holding_bars": outcome.holding_bars,
                "mfe_pct": outcome.mfe_pct,
                "mae_pct": outcome.mae_pct,
                "is_tp": 1 if outcome.outcome == "tp" else 0 if outcome.outcome in {"sl", "timeout", "ambiguous"} else np.nan,
                "is_sl": 1 if outcome.outcome == "sl" else 0 if outcome.outcome in {"tp", "timeout", "ambiguous"} else np.nan,
                "is_resolved": 1 if outcome.outcome in {"tp", "sl"} else 0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Strategy context from the signal stream itself
# --------------------------------------------------------------------------------------
def compute_open_trade_counts(signals: pd.DataFrame) -> pd.Series:
    entries = pd.to_datetime(signals["entry_time"], errors="coerce").dropna().sort_values().to_numpy()
    exits = pd.to_datetime(signals["exit_time"], errors="coerce").dropna().sort_values().to_numpy()
    times = signals["open_time"].to_numpy()

    counts = []
    for t in times:
        opened = np.searchsorted(entries, t, side="left")
        closed = np.searchsorted(exits, t, side="left")
        counts.append(max(0, opened - closed))
    return pd.Series(counts, index=signals.index, dtype=float)


def compute_signal_context(signals: pd.DataFrame) -> pd.DataFrame:
    df = signals.sort_values(["open_time", "symbol"]).reset_index(drop=True).copy()

    overall_windows = {
        "signal_ctx_signals_last_1h": pd.Timedelta(hours=1),
        "signal_ctx_signals_last_4h": pd.Timedelta(hours=4),
        "signal_ctx_signals_last_12h": pd.Timedelta(hours=12),
    }
    window_deques = {name: deque() for name in overall_windows}

    same_symbol_24h = {}
    uniq_4h_events = deque()
    uniq_4h_counter = Counter()

    resolved_closed = (
        df[df["trade_outcome"].isin(["tp", "sl"])][["exit_time", "trade_outcome", "trade_pnl_pct"]]
        .dropna(subset=["exit_time"])
        .sort_values("exit_time")
        .reset_index(drop=True)
    )

    close_ptr = 0
    closed_24h = deque()
    closed_5 = deque(maxlen=5)
    current_sl_streak = 0
    current_tp_streak = 0

    rows = []
    for row in df.itertuples():
        t = row.open_time
        symbol = row.symbol

        # Advance closed trades that are known by time t.
        while close_ptr < len(resolved_closed) and resolved_closed.iloc[close_ptr]["exit_time"] < t:
            outcome = resolved_closed.iloc[close_ptr]["trade_outcome"]
            pnl = float(resolved_closed.iloc[close_ptr]["trade_pnl_pct"])
            exit_time = resolved_closed.iloc[close_ptr]["exit_time"]
            closed_24h.append((exit_time, outcome, pnl))
            closed_5.append((exit_time, outcome, pnl))

            if outcome == "sl":
                current_sl_streak += 1
                current_tp_streak = 0
            elif outcome == "tp":
                current_tp_streak += 1
                current_sl_streak = 0

            close_ptr += 1

        # Remove outdated trades from 24h window.
        while closed_24h and closed_24h[0][0] < t - pd.Timedelta(hours=24):
            closed_24h.popleft()

        # Remove outdated overall signal window items.
        for name, delta in overall_windows.items():
            dq = window_deques[name]
            while dq and dq[0] < t - delta:
                dq.popleft()

        # Same-symbol 24h window.
        symdq = same_symbol_24h.setdefault(symbol, deque())
        while symdq and symdq[0] < t - pd.Timedelta(hours=24):
            symdq.popleft()

        # Unique symbols in last 4h.
        while uniq_4h_events and uniq_4h_events[0][0] < t - pd.Timedelta(hours=4):
            old_t, old_sym = uniq_4h_events.popleft()
            uniq_4h_counter[old_sym] -= 1
            if uniq_4h_counter[old_sym] <= 0:
                del uniq_4h_counter[old_sym]

        closed_24h_outcomes = [x[1] for x in closed_24h]
        closed_24h_pnls = [x[2] for x in closed_24h]
        closed_5_outcomes = [x[1] for x in closed_5]
        closed_5_pnls = [x[2] for x in closed_5]

        row_ctx = {
            "signal_id": row.signal_id,
            "signal_ctx_signals_last_1h": len(window_deques["signal_ctx_signals_last_1h"]),
            "signal_ctx_signals_last_4h": len(window_deques["signal_ctx_signals_last_4h"]),
            "signal_ctx_signals_last_12h": len(window_deques["signal_ctx_signals_last_12h"]),
            "signal_ctx_same_symbol_last_24h": len(symdq),
            "signal_ctx_unique_symbols_last_4h": len(uniq_4h_counter),
            "signal_ctx_closed_last_24h": len(closed_24h),
            "signal_ctx_resolved_sl_rate_last_24h": (
                np.mean(np.array(closed_24h_outcomes) == "sl") if closed_24h_outcomes else np.nan
            ),
            "signal_ctx_resolved_pnl_sum_last_24h": float(np.sum(closed_24h_pnls)) if closed_24h_pnls else np.nan,
            "signal_ctx_closed_last_5": len(closed_5),
            "signal_ctx_resolved_sl_rate_last_5": (
                np.mean(np.array(closed_5_outcomes) == "sl") if closed_5_outcomes else np.nan
            ),
            "signal_ctx_resolved_pnl_sum_last_5": float(np.sum(closed_5_pnls)) if closed_5_pnls else np.nan,
            "signal_ctx_prev_closed_sl_streak": current_sl_streak,
            "signal_ctx_prev_closed_tp_streak": current_tp_streak,
            "signal_ctx_last_closed_is_sl": (
                1 if closed_5_outcomes and closed_5_outcomes[-1] == "sl" else 0 if closed_5_outcomes else np.nan
            ),
        }
        rows.append(row_ctx)

        # Add current signal after recording state.
        for dq in window_deques.values():
            dq.append(t)
        symdq.append(t)
        uniq_4h_events.append((t, symbol))
        uniq_4h_counter[symbol] += 1

    result = pd.DataFrame(rows)
    result["signal_ctx_open_trades_now"] = compute_open_trade_counts(df)
    return result


# --------------------------------------------------------------------------------------
# Candidate gate targets: "should we pause because a loss regime is starting?"
# --------------------------------------------------------------------------------------
def build_future_targets(signals: pd.DataFrame) -> pd.DataFrame:
    df = signals.sort_values(["open_time", "symbol"]).reset_index(drop=True).copy()
    resolved = df[df["trade_outcome"].isin(["tp", "sl"])][
        ["open_time", "trade_outcome", "trade_pnl_pct"]
    ].reset_index(drop=True)

    rows = []
    for row in df.itertuples():
        t = row.open_time
        future_resolved = resolved[resolved["open_time"] >= t].copy()

        feature_row = {"signal_id": row.signal_id}

        for n in NEXT_SIGNAL_WINDOWS:
            fr = future_resolved.head(n)
            if len(fr) < n:
                feature_row[f"future_resolved_count_next_{n}"] = len(fr)
                feature_row[f"future_sl_rate_next_{n}"] = np.nan
                feature_row[f"future_pnl_sum_next_{n}"] = np.nan
                feature_row[f"target_bad_next_{n}_signals"] = np.nan
            else:
                sl_rate = float((fr["trade_outcome"] == "sl").mean())
                pnl_sum = float(fr["trade_pnl_pct"].sum())
                feature_row[f"future_resolved_count_next_{n}"] = len(fr)
                feature_row[f"future_sl_rate_next_{n}"] = sl_rate
                feature_row[f"future_pnl_sum_next_{n}"] = pnl_sum
                feature_row[f"target_bad_next_{n}_signals"] = int(sl_rate >= 0.60 and pnl_sum < 0)

        for hours in NEXT_TIME_WINDOWS_HOURS:
            fr = resolved[(resolved["open_time"] >= t) & (resolved["open_time"] < t + pd.Timedelta(hours=hours))].copy()
            min_needed = MIN_RESOLVED_SIGNALS_FOR_TIME_TARGET[hours]
            if len(fr) < min_needed:
                feature_row[f"future_resolved_count_next_{hours}h"] = len(fr)
                feature_row[f"future_sl_rate_next_{hours}h"] = np.nan
                feature_row[f"future_pnl_sum_next_{hours}h"] = np.nan
                feature_row[f"target_bad_next_{hours}h"] = np.nan
            else:
                sl_rate = float((fr["trade_outcome"] == "sl").mean())
                pnl_sum = float(fr["trade_pnl_pct"].sum())
                feature_row[f"future_resolved_count_next_{hours}h"] = len(fr)
                feature_row[f"future_sl_rate_next_{hours}h"] = sl_rate
                feature_row[f"future_pnl_sum_next_{hours}h"] = pnl_sum
                feature_row[f"target_bad_next_{hours}h"] = int(sl_rate >= 0.67 and pnl_sum < 0)

        rows.append(feature_row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Aggregations for interpretable inspection
# --------------------------------------------------------------------------------------
WINDOW_CONTEXT_COLS = [
    "btc_ret_16",
    "eth_ret_16",
    "breadth_pos_16",
    "breadth_mean_ret_16",
    "breadth_share_gt_5pct_16",
    "breadth_share_near_high_96",
    "breadth_share_vol_spike_20",
    "signal_ctx_signals_last_4h",
    "signal_ctx_open_trades_now",
    "signal_ctx_resolved_sl_rate_last_24h",
    "token_ret_16",
    "token_minus_btc_ret_16",
    "token_vol_ratio_20",
]


def aggregate_time_windows(signals: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = signals.copy()
    df["window_start"] = df["open_time"].dt.floor(freq)

    agg_dict = {
        "signal_id": "count",
        "symbol": pd.Series.nunique,
        "is_resolved": "sum",
        "is_tp": "sum",
        "is_sl": "sum",
        "trade_pnl_pct": "sum",
        "holding_bars": "mean",
        "mfe_pct": "mean",
        "mae_pct": "mean",
    }
    for col in WINDOW_CONTEXT_COLS:
        if col in df.columns:
            agg_dict[col] = "mean"

    grouped = df.groupby("window_start").agg(agg_dict).rename(
        columns={
            "signal_id": "signals",
            "symbol": "unique_symbols",
            "is_resolved": "resolved",
            "is_tp": "tp",
            "is_sl": "sl",
            "trade_pnl_pct": "pnl_sum",
            "holding_bars": "avg_holding_bars",
            "mfe_pct": "avg_mfe_pct",
            "mae_pct": "avg_mae_pct",
        }
    )
    grouped = grouped.reset_index()

    grouped["timeout"] = df.groupby("window_start")["trade_outcome"].apply(lambda s: int((s == "timeout").sum())).values
    grouped["ambiguous"] = df.groupby("window_start")["trade_outcome"].apply(lambda s: int((s == "ambiguous").sum())).values
    grouped["sl_rate_resolved"] = grouped["sl"] / grouped["resolved"].replace(0, np.nan)
    grouped["tp_rate_resolved"] = grouped["tp"] / grouped["resolved"].replace(0, np.nan)
    grouped["pnl_per_signal"] = grouped["pnl_sum"] / grouped["signals"].replace(0, np.nan)

    return grouped.sort_values("window_start").reset_index(drop=True)


def build_streaks(signals: pd.DataFrame) -> pd.DataFrame:
    df = signals[signals["trade_outcome"].isin(["tp", "sl"])].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("open_time").reset_index(drop=True)
    streak_ids = []
    current = 0
    prev_outcome = None
    prev_time = None

    for row in df.itertuples():
        new_streak = False
        if prev_outcome is None:
            new_streak = True
        elif row.trade_outcome != prev_outcome:
            new_streak = True
        elif (row.open_time - prev_time) > pd.Timedelta(hours=STREAK_BREAK_GAP_HOURS):
            new_streak = True

        if new_streak:
            current += 1

        streak_ids.append(current)
        prev_outcome = row.trade_outcome
        prev_time = row.open_time

    df["streak_id"] = streak_ids

    agg_dict = {
        "open_time": ["min", "max", "count"],
        "trade_pnl_pct": "sum",
        "symbol": pd.Series.nunique,
    }
    for col in WINDOW_CONTEXT_COLS:
        if col in df.columns:
            agg_dict[col] = "mean"

    streaks = df.groupby(["streak_id", "trade_outcome"]).agg(agg_dict)
    streaks.columns = ["_".join([c for c in col if c]).strip("_") for col in streaks.columns.values]
    streaks = streaks.reset_index()

    streaks = streaks.rename(
        columns={
            "trade_outcome": "streak_outcome",
            "open_time_min": "start_time",
            "open_time_max": "end_time",
            "open_time_count": "signals",
            "trade_pnl_pct_sum": "pnl_sum",
            "symbol_nunique": "unique_symbols",
        }
    )
    streaks["duration_hours"] = (
        (pd.to_datetime(streaks["end_time"]) - pd.to_datetime(streaks["start_time"])).dt.total_seconds() / 3600.0
    )
    return streaks.sort_values(["signals", "start_time"], ascending=[False, True]).reset_index(drop=True)


def build_focus_windows(signals: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    years = sorted(signals["open_time"].dt.year.dropna().unique().tolist())
    windows = []
    for year in years:
        for name, start_md, end_md in FOCUS_MONTH_DAY_WINDOWS:
            start = pd.Timestamp(f"{year}-{start_md}")
            end = pd.Timestamp(f"{year}-{end_md}")
            window_mask = (signals["open_time"] >= start - pd.Timedelta(hours=12)) & (
                signals["open_time"] <= end + pd.Timedelta(hours=12)
            )
            if window_mask.any():
                windows.append((f"{name}_{year}", start, end))
    return windows


def summarize_slice(df: pd.DataFrame, label: str) -> dict:
    resolved = df[df["trade_outcome"].isin(["tp", "sl"])]
    result = {
        "label": label,
        "signals": len(df),
        "resolved": len(resolved),
        "tp": int((df["trade_outcome"] == "tp").sum()),
        "sl": int((df["trade_outcome"] == "sl").sum()),
        "timeout": int((df["trade_outcome"] == "timeout").sum()),
        "ambiguous": int((df["trade_outcome"] == "ambiguous").sum()),
        "pnl_sum": float(df["trade_pnl_pct"].sum()) if "trade_pnl_pct" in df.columns else np.nan,
        "sl_rate_resolved": float((resolved["trade_outcome"] == "sl").mean()) if len(resolved) else np.nan,
        "avg_holding_bars": float(resolved["holding_bars"].mean()) if len(resolved) else np.nan,
    }
    for col in WINDOW_CONTEXT_COLS:
        if col in df.columns:
            result[col] = float(df[col].mean())
    return result


# --------------------------------------------------------------------------------------
# Feature ranking / exploratory modeling
# --------------------------------------------------------------------------------------
EXCLUDE_FEATURES = {
    "signal_id",
    "symbol",
    "open_time",
    "source_file",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "trade_outcome",
    "trade_pnl_pct",
    "trade_pnl_pct_best",
    "trade_pnl_pct_worst",
    "is_tp",
    "is_sl",
    "is_resolved",
    "pump_la_type",
    "timeframe",
    "event_id",
    "target",
}


def feature_family(name: str) -> str:
    if name.startswith("breadth_"):
        return "market_breadth"
    if name.startswith("btc_"):
        return "btc_context"
    if name.startswith("eth_"):
        return "eth_context"
    if name.startswith("signal_ctx_"):
        return "strategy_context"
    if name.startswith("token_"):
        return "token_interpretable"
    if name.startswith("liq_") or "pdh" in name or "pwh" in name or "eqh" in name:
        return "liquidity"
    if (
        name.startswith("runup")
        or name.startswith("pump_")
        or name.startswith("predump")
        or name.startswith("strong_cond")
        or name.startswith("vol_fade")
        or name.startswith("rsi_fade")
        or name.startswith("macd_fade")
    ):
        return "pump_detector"
    if (
        name.startswith("ret_")
        or name.startswith("cum_ret")
        or name.startswith("rsi")
        or name.startswith("mfi")
        or name.startswith("macd")
        or name.startswith("atr")
        or name.startswith("bb_")
        or name.startswith("vwap")
        or name.startswith("drawdown")
        or name.startswith("volume_")
        or name.startswith("vol_ratio")
        or name.startswith("close_pos")
        or name.startswith("wick")
        or name.startswith("body_ratio")
        or name.startswith("count_red")
        or name.startswith("max_upper_wick")
    ):
        return "current_model_numeric"
    return "other"


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in EXCLUDE_FEATURES
        and not c.startswith("future_")
        and not c.startswith("target_")
    ]
    return clean_constant_columns(df, numeric_cols)


def single_feature_rankings(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    work = df.dropna(subset=[target_col]).copy()
    if work.empty or work[target_col].nunique() < 2:
        return pd.DataFrame()

    rows = []
    y = work[target_col].astype(int)

    for col in feature_cols:
        x = work[col]
        mask = x.notna() & y.notna()
        if mask.sum() < MIN_FEATURE_ROWS:
            continue

        xs = x[mask]
        ys = y[mask]

        if xs.nunique(dropna=True) <= 1 or ys.nunique() < 2:
            continue

        try:
            raw_auc = roc_auc_score(ys, xs)
            direction = "high->risk" if raw_auc >= 0.5 else "low->risk"
            auc = max(raw_auc, 1 - raw_auc)
        except Exception:
            continue

        pos = xs[ys == 1]
        neg = xs[ys == 0]

        rows.append(
            {
                "feature": col,
                "family": feature_family(col),
                "rows": int(mask.sum()),
                "single_auc": float(auc),
                "direction": direction,
                "mean_pos": float(pos.mean()),
                "mean_neg": float(neg.mean()),
                "median_pos": float(pos.median()),
                "median_neg": float(neg.median()),
                "delta_mean": float(pos.mean() - neg.mean()),
            }
        )

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows).sort_values(["single_auc", "rows"], ascending=[False, False]).reset_index(drop=True)
    return ranked


def fit_quick_catboost(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    work = df.dropna(subset=[target_col]).sort_values("open_time").copy()
    if work.empty or work[target_col].nunique() < 2:
        return {}, pd.DataFrame()

    use_cols = clean_constant_columns(work, feature_cols)
    if not use_cols:
        return {}, pd.DataFrame()

    split_idx = max(int(len(work) * 0.75), 1)
    if split_idx >= len(work):
        return {}, pd.DataFrame()

    train = work.iloc[:split_idx].copy()
    test = work.iloc[split_idx:].copy()

    # Make sure both splits contain both classes, otherwise shrink/grow boundary a bit.
    if train[target_col].nunique() < 2 or test[target_col].nunique() < 2:
        return {}, pd.DataFrame()

    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(train[use_cols], train[target_col], eval_set=(test[use_cols], test[target_col]), use_best_model=True)

    pred = model.predict_proba(test[use_cols])[:, 1]
    metrics = {
        "target": target_col,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": float(train[target_col].mean()),
        "test_positive_rate": float(test[target_col].mean()),
        "test_auc": float(roc_auc_score(test[target_col], pred)),
        "test_pr_auc": float(average_precision_score(test[target_col], pred)),
        "used_features": len(use_cols),
        "best_iteration": int(model.get_best_iteration()),
    }

    importances = model.get_feature_importance(type="PredictionValuesChange")
    imp_df = pd.DataFrame({"feature": use_cols, "importance": importances})
    imp_df["family"] = imp_df["feature"].map(feature_family)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, imp_df


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------
def enrich_relative_features(signals: pd.DataFrame) -> pd.DataFrame:
    df = signals.copy()
    for w in (4, 16, 96):
        token_col = f"token_ret_{w}"
        btc_col = f"btc_ret_{w}"
        eth_col = f"eth_ret_{w}"
        if token_col in df.columns and btc_col in df.columns:
            df[f"token_minus_btc_ret_{w}"] = df[token_col] - df[btc_col]
        if token_col in df.columns and eth_col in df.columns:
            df[f"token_minus_eth_ret_{w}"] = df[token_col] - df[eth_col]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_summary_markdown(
    signals: pd.DataFrame,
    focus_summary: pd.DataFrame,
    windows_6h: pd.DataFrame,
    streaks: pd.DataFrame,
    target_rankings: dict[str, pd.DataFrame],
    model_metrics: dict[str, dict],
    model_family_importance: dict[str, pd.DataFrame],
    signal_files: list[str],
) -> str:
    overall = summarize_slice(signals, "overall")
    best_6h = windows_6h.sort_values("pnl_sum", ascending=False).head(8)
    worst_6h = windows_6h.sort_values("pnl_sum", ascending=True).head(8)

    lines = []
    lines.append("# Pump-end regime analysis")
    lines.append("")
    lines.append("## Inputs / assumptions")
    lines.append("")
    lines.append(f"- signal files: {', '.join(signal_files)}")
    lines.append(f"- entry bar shift: {ENTRY_BAR_SHIFT} (0 = same bar open, 1 = next bar open)")
    lines.append(f"- TP: {TP_PCT:.2%}")
    lines.append(f"- SL: {SL_PCT:.2%}")
    lines.append(f"- max hold: {MAX_HOLD_BARS} bars ({MAX_HOLD_BARS * 15 / 60:.1f}h)")
    lines.append(f"- current model feature snapshot: window_bars={WINDOW_BARS}, warmup_bars={WARMUP_BARS}, feature_set={FEATURE_SET}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame([overall])))
    lines.append("")
    lines.append("## Focus windows from the brief")
    lines.append("")
    lines.append(markdown_table(focus_summary))
    lines.append("")
    lines.append("## Worst 6h windows")
    lines.append("")
    lines.append(markdown_table(safe_subset(worst_6h, ["window_start", "signals", "resolved", "tp", "sl", "timeout", "ambiguous", "pnl_sum", "sl_rate_resolved", "btc_ret_16", "breadth_share_gt_5pct_16"])))
    lines.append("")
    lines.append("## Best 6h windows")
    lines.append("")
    lines.append(markdown_table(safe_subset(best_6h, ["window_start", "signals", "resolved", "tp", "sl", "timeout", "ambiguous", "pnl_sum", "sl_rate_resolved", "btc_ret_16", "breadth_share_gt_5pct_16"])))
    lines.append("")
    lines.append("## Longest loss streaks")
    lines.append("")
    loss_streaks = streaks[streaks["streak_outcome"] == "sl"].head(10)
    lines.append(markdown_table(safe_subset(loss_streaks, ["start_time", "end_time", "signals", "unique_symbols", "pnl_sum", "duration_hours", "btc_ret_16_mean", "breadth_share_gt_5pct_16_mean"])) if not loss_streaks.empty else "_empty_")
    lines.append("")
    lines.append("## Longest win streaks")
    lines.append("")
    win_streaks = streaks[streaks["streak_outcome"] == "tp"].head(10)
    lines.append(markdown_table(safe_subset(win_streaks, ["start_time", "end_time", "signals", "unique_symbols", "pnl_sum", "duration_hours", "btc_ret_16_mean", "breadth_share_gt_5pct_16_mean"])) if not win_streaks.empty else "_empty_")
    lines.append("")

    for target_name, ranking_df in target_rankings.items():
        lines.append(f"## Top univariate separators for {target_name}")
        lines.append("")
        lines.append(markdown_table(safe_subset(ranking_df, ["feature", "family", "single_auc", "direction", "rows", "mean_pos", "mean_neg"])))
        lines.append("")
        metrics = model_metrics.get(target_name, {})
        if metrics:
            lines.append(f"### Quick CatBoost check for {target_name}")
            lines.append("")
            lines.append(markdown_table(pd.DataFrame([metrics])))
            fam_df = model_family_importance.get(target_name, pd.DataFrame())
            if not fam_df.empty:
                lines.append("")
                lines.append("Top feature families:")
                lines.append("")
                lines.append(markdown_table(fam_df))
            lines.append("")

    lines.append("## Reading guide")
    lines.append("")
    lines.append("- `signals_enriched.parquet` — full signal-level dataset for follow-up work.")
    lines.append("- `focus_windows/*.csv` — raw rows for the two periods you described.")
    lines.append("- `window_summary_6h.csv` — best/worst regime blocks.")
    lines.append("- `streaks.csv` — consecutive TP/SL sequences.")
    lines.append("- `feature_ranking_*.csv` — fast separators before we design the gate model.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pump-end signal regimes and loss clusters")
    parser.add_argument("--ch-db", required=True, help="ClickHouse DSN, e.g. http://user:pass@host:8123/db")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    focus_dir = OUTPUT_DIR / "focus_windows"
    focus_dir.mkdir(parents=True, exist_ok=True)

    log("loading signal files")
    signals = load_signals()
    signal_files = sorted(signals["source_file"].dropna().unique().tolist())

    if signals.empty:
        raise RuntimeError("No signals after parsing input files")

    log(f"signals loaded: {len(signals)} rows, {signals['symbol'].nunique()} symbols")

    loader = DataLoader(args.ch_db)

    buffer_before_bars = WARMUP_BARS + WINDOW_BARS + DEFAULT_PUMP_PARAMS.liquidity_window_bars + 21 + HIGH_LOOKBACK
    range_start = signals["open_time"].min() - pd.Timedelta(minutes=15 * buffer_before_bars)
    range_end = signals["open_time"].max() + pd.Timedelta(minutes=15 * (MAX_HOLD_BARS + 2))

    # Load signaled symbols + reference markets once.
    signal_symbols = sorted(set(signals["symbol"].unique().tolist()) | set(REFERENCE_SYMBOLS))
    log(f"loading candles for signaled symbols and refs: {len(signal_symbols)} symbols")
    symbol_candles = load_candles_dict(
        loader,
        signal_symbols,
        range_start,
        range_end,
        batch_size=SIGNAL_CANDLES_BATCH_SIZE,
        label="signal/ref candles",
    )

    btc_candles = symbol_candles.get("BTCUSDT")
    eth_candles = symbol_candles.get("ETHUSDT")

    # Trade outcomes.
    log("simulating trade outcomes")
    trade_outcomes = compute_trade_outcomes(signals, symbol_candles)
    signals = signals.merge(trade_outcomes, on="signal_id", how="left")

    # Context from the current model feature builder.
    log("building current-model snapshot features")
    snapshot_features = build_current_model_snapshot_features(signals, symbol_candles, btc_candles)
    if not snapshot_features.empty:
        signals = signals.merge(
            snapshot_features.drop_duplicates(subset=["signal_id"]),
            on=["signal_id", "symbol", "open_time"],
            how="left",
            suffixes=("", "_model"),
        )

    # Interpretable token + BTC + ETH features.
    log("merging token-level interpretable context")
    signals = merge_symbol_context(signals, symbol_candles, "token_")
    log("merging BTC context")
    signals = merge_reference_context(signals, btc_candles, "btc_")
    log("merging ETH context")
    signals = merge_reference_context(signals, eth_candles, "eth_")
    signals = enrich_relative_features(signals)

    # Broad market breadth.
    log("computing broad-market breadth")
    breadth_df = compute_market_breadth(loader, range_start, range_end)
    if not breadth_df.empty:
        signals = signals.merge(breadth_df.rename(columns={"bucket": "open_time"}), on="open_time", how="left")

    # Strategy context from recent signal stream.
    log("computing signal-stream context")
    signal_ctx = compute_signal_context(signals)
    signals = signals.merge(signal_ctx, on="signal_id", how="left")

    # Candidate future targets for the gate model.
    log("building future regime targets")
    future_targets = build_future_targets(signals)
    signals = signals.merge(future_targets, on="signal_id", how="left")

    # Output raw enriched dataset.
    signals = signals.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    save_parquet(signals, OUTPUT_DIR / "signals_enriched.parquet")
    save_csv(signals, OUTPUT_DIR / "signals_enriched.csv")

    # Aggregated windows and streaks.
    log("building aggregated regime views")
    window_3h = aggregate_time_windows(signals, "3h")
    window_6h = aggregate_time_windows(signals, "6h")
    window_12h = aggregate_time_windows(signals, "12h")
    save_csv(window_3h, OUTPUT_DIR / "window_summary_3h.csv")
    save_csv(window_6h, OUTPUT_DIR / "window_summary_6h.csv")
    save_csv(window_12h, OUTPUT_DIR / "window_summary_12h.csv")

    streaks = build_streaks(signals)
    save_csv(streaks, OUTPUT_DIR / "streaks.csv")

    # Focus windows from the brief.
    focus_windows = build_focus_windows(signals)
    focus_rows = []
    for label, start, end in focus_windows:
        sliced = signals[(signals["open_time"] >= start) & (signals["open_time"] <= end)].copy()
        focus_rows.append(summarize_slice(sliced, label))
        save_csv(sliced, focus_dir / f"{label}.csv")
    focus_summary = pd.DataFrame(focus_rows).sort_values("label") if focus_rows else pd.DataFrame()
    if not focus_summary.empty:
        save_csv(focus_summary, OUTPUT_DIR / "focus_window_summary.csv")

    # Exploratory rankings / quick models.
    log("ranking candidate features")
    feature_cols = candidate_feature_columns(signals)
    target_defs = [
        "is_sl",
        "target_bad_next_5_signals",
        "target_bad_next_12h",
    ]

    target_rankings: dict[str, pd.DataFrame] = {}
    model_metrics: dict[str, dict] = {}
    model_family_importance: dict[str, pd.DataFrame] = {}

    for target in target_defs:
        if target not in signals.columns:
            continue

        rankings = single_feature_rankings(signals, target, feature_cols)
        if rankings.empty:
            continue

        target_rankings[target] = rankings.head(25).copy()
        save_csv(rankings, OUTPUT_DIR / f"feature_ranking_{target}.csv")

        selected_features = rankings["feature"].head(MAX_MODEL_FEATURES).tolist()
        metrics, importance_df = fit_quick_catboost(signals, target, selected_features)

        if metrics:
            model_metrics[target] = metrics
            save_csv(pd.DataFrame([metrics]), OUTPUT_DIR / f"quick_model_metrics_{target}.csv")

        if not importance_df.empty:
            save_csv(importance_df, OUTPUT_DIR / f"quick_model_feature_importance_{target}.csv")
            fam = (
                importance_df.groupby("family", as_index=False)["importance"].sum()
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            model_family_importance[target] = fam
            save_csv(fam, OUTPUT_DIR / f"quick_model_family_importance_{target}.csv")

    # Summary markdown.
    summary_md = build_summary_markdown(
        signals=signals,
        focus_summary=focus_summary if not focus_summary.empty else pd.DataFrame([{"label": "no focus windows found in input range"}]),
        windows_6h=window_6h,
        streaks=streaks,
        target_rankings=target_rankings,
        model_metrics=model_metrics,
        model_family_importance=model_family_importance,
        signal_files=signal_files,
    )
    (OUTPUT_DIR / "analysis_summary.md").write_text(summary_md, encoding="utf-8")

    # Lightweight JSON metadata for easier sharing back.
    meta = {
        "signals": int(len(signals)),
        "symbols": int(signals["symbol"].nunique()),
        "date_from": str(signals["open_time"].min()),
        "date_to": str(signals["open_time"].max()),
        "outputs": sorted([p.name for p in OUTPUT_DIR.glob("*") if p.is_file()]),
        "focus_windows": [row["label"] for row in focus_rows] if focus_rows else [],
    }
    (OUTPUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    resolved = signals[signals["trade_outcome"].isin(["tp", "sl"])]
    log("done")
    log(f"output dir: {OUTPUT_DIR}")
    log(
        "overall: "
        f"signals={len(signals)} "
        f"resolved={len(resolved)} "
        f"tp={(signals['trade_outcome'] == 'tp').sum()} "
        f"sl={(signals['trade_outcome'] == 'sl').sum()} "
        f"timeout={(signals['trade_outcome'] == 'timeout').sum()} "
        f"ambiguous={(signals['trade_outcome'] == 'ambiguous').sum()}"
    )


if __name__ == "__main__":
    main()
