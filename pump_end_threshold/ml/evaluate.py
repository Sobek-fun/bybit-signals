import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)

from pump_end_threshold.ml.predict import build_pending_turn_down_decision_table, extract_signals

PRIMARY_HORIZON_BARS = 32
PRIMARY_SQUEEZE_PCT = 0.02
PRIMARY_PULLBACK_PCT = 0.03
DEFAULT_SIGNAL_QUALITY_HORIZONS = [16, 32]


def compute_event_level_metrics(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None,
        abstain_margin: float = 0.0
) -> dict:
    decision_df = build_pending_turn_down_decision_table(
        predictions_df=predictions_df,
        threshold=threshold,
        signal_rule=signal_rule,
        min_pending_bars=min_pending_bars,
        drop_delta=drop_delta,
        abstain_margin=abstain_margin,
        event_data=event_data,
    )

    if decision_df.empty:
        return {
            'n_events': 0,
            'hit0': 0,
            'hit0_rate': 0,
            'hit0_or_hit1': 0,
            'hit0_or_hit1_rate': 0,
            'early': 0,
            'early_rate': 0,
            'late': 0,
            'late_rate': 0,
            'miss': 0,
            'miss_rate': 0,
            'avg_pred_offset': None,
            'median_pred_offset': None,
            'n_b': 0,
            'false_positive_b': 0,
            'true_negative_b': 0,
            'fp_b_rate': 0,
        }

    b_df = decision_df[decision_df['event_type'] == 'B']
    a_df = decision_df[decision_df['event_type'] != 'B']
    a_triggered = a_df[a_df['triggered']]
    offsets = a_triggered['signal_offset'].dropna().to_numpy()

    hit0 = int((offsets == 0).sum())
    hit1 = int((offsets == 1).sum())
    early = int((offsets < 0).sum())
    late = int((offsets > 1).sum())
    miss = int((~a_df['triggered']).sum())
    false_positive_b = int((b_df['triggered']).sum())
    true_negative_b = int((~b_df['triggered']).sum())
    n_b = int(len(b_df))
    n_a = int(len(a_df))
    n_events = n_a

    return {
        'n_events': n_events,
        'hit0': hit0,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hit0_or_hit1': hit0 + hit1,
        'hit0_or_hit1_rate': (hit0 + hit1) / n_events if n_events > 0 else 0,
        'early': early,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late': late,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss': miss,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'avg_pred_offset': float(np.mean(offsets)) if len(offsets) > 0 else None,
        'median_pred_offset': float(np.median(offsets)) if len(offsets) > 0 else None,
        'n_b': n_b,
        'false_positive_b': false_positive_b,
        'true_negative_b': true_negative_b,
        'fp_b_rate': false_positive_b / n_b if n_b > 0 else 0
    }


def compute_point_level_metrics(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    y_true = predictions_df['y'].values
    y_prob = predictions_df['p_end'].values

    if len(y_true) == 0:
        return {
            'pr_auc': None,
            'roc_auc': None,
            'precision_at_threshold': 0,
            'recall_at_threshold': 0,
            'threshold_used': threshold
        }

    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision_at_threshold': precision,
        'recall_at_threshold': recall,
        'threshold_used': threshold
    }


def compute_trade_quality_metrics(
        signals_df: pd.DataFrame,
        candles_loader,
        horizons: list = None
) -> dict:
    if horizons is None:
        horizons = [16, 32]

    if signals_df.empty:
        return {f'mfe_short_{h}': {} for h in horizons} | {f'mae_short_{h}': {} for h in horizons}

    signals_df = signals_df.copy()
    signals_df['open_time'] = pd.to_datetime(signals_df['open_time'])

    results = {}
    max_horizon = max(horizons)

    grouped = signals_df.groupby('symbol')

    symbol_candles = {}
    for symbol, group in grouped:
        min_time = group['open_time'].min()
        max_time = group['open_time'].max() + pd.Timedelta(minutes=(max_horizon + 1) * 15)

        df = candles_loader.load_candles_range(symbol, min_time, max_time)
        if not df.empty:
            symbol_candles[symbol] = df

    for h in horizons:
        mfe_values = []
        mae_values = []

        for row in signals_df.itertuples():
            symbol = row.symbol
            entry_time = row.open_time

            if symbol not in symbol_candles:
                continue

            df = symbol_candles[symbol]

            if entry_time not in df.index:
                continue

            entry_idx = df.index.get_loc(entry_time)

            if entry_idx + h >= len(df):
                continue

            entry_open = df.iloc[entry_idx]['open']
            if entry_open <= 0:
                continue

            future_df = df.iloc[entry_idx + 1:entry_idx + h + 1]

            if future_df.empty:
                continue

            min_low = future_df['low'].min()
            max_high = future_df['high'].max()

            mfe = (entry_open - min_low) / entry_open
            mae = (max_high - entry_open) / entry_open

            mfe_values.append(mfe)
            mae_values.append(mae)

        if mfe_values:
            results[f'mfe_short_{h}'] = {
                'mean': np.mean(mfe_values),
                'median': np.median(mfe_values),
                'p25': np.percentile(mfe_values, 25),
                'p75': np.percentile(mfe_values, 75),
                'min': np.min(mfe_values),
                'max': np.max(mfe_values),
                'pct_above_2pct': np.mean(np.array(mfe_values) >= 0.02),
                'count': len(mfe_values)
            }
        else:
            results[f'mfe_short_{h}'] = {}

        if mae_values:
            results[f'mae_short_{h}'] = {
                'mean': np.mean(mae_values),
                'median': np.median(mae_values),
                'p25': np.percentile(mae_values, 25),
                'p75': np.percentile(mae_values, 75),
                'min': np.min(mae_values),
                'max': np.max(mae_values),
                'count': len(mae_values)
            }
        else:
            results[f'mae_short_{h}'] = {}

    return results


def compute_trade_quality_score(trade_metrics: dict, horizon: int = 32) -> float:
    mfe_key = f'mfe_short_{horizon}'
    mae_key = f'mae_short_{horizon}'

    if mfe_key not in trade_metrics or not trade_metrics[mfe_key]:
        return -np.inf

    mfe_stats = trade_metrics[mfe_key]
    mae_stats = trade_metrics.get(mae_key, {})

    mfe_median = mfe_stats.get('median', 0)
    mfe_pct_above_2pct = mfe_stats.get('pct_above_2pct', 0)
    mae_median = mae_stats.get('median', 0.1)

    score = mfe_median * 100 + mfe_pct_above_2pct * 50 - mae_median * 30

    return score


def _pair_key(squeeze_threshold: float, pullback_threshold: float) -> str:
    squeeze_pct = int(round(squeeze_threshold * 100))
    pullback_pct = int(round(pullback_threshold * 100))
    return f"{squeeze_pct}_{pullback_pct}"


def _empty_signal_quality_metrics(
        horizon: int = PRIMARY_HORIZON_BARS,
        squeeze_threshold: float = PRIMARY_SQUEEZE_PCT,
        pullback_threshold: float = PRIMARY_PULLBACK_PCT
) -> dict:
    pair = _pair_key(squeeze_threshold, pullback_threshold)
    h_tag = f"h{horizon}"
    return {
        f"squeeze_median_{h_tag}": np.nan,
        f"squeeze_mean_{h_tag}": np.nan,
        f"squeeze_p75_{h_tag}": np.nan,
        f"pullback_median_{h_tag}": np.nan,
        f"pullback_mean_{h_tag}": np.nan,
        f"pullback_p75_{h_tag}": np.nan,
        f"net_edge_median_{h_tag}": np.nan,
        f"net_edge_mean_{h_tag}": np.nan,
        f"clean_{pair}_count_{h_tag}": 0,
        f"clean_{pair}_share_{h_tag}": 0.0,
        f"dirty_retrace_{pair}_count_{h_tag}": 0,
        f"dirty_retrace_{pair}_share_{h_tag}": 0.0,
        f"clean_no_pullback_{pair}_count_{h_tag}": 0,
        f"clean_no_pullback_{pair}_share_{h_tag}": 0.0,
        f"dirty_no_pullback_{pair}_count_{h_tag}": 0,
        f"dirty_no_pullback_{pair}_share_{h_tag}": 0.0,
        f"clean_to_dirty_failure_ratio_{pair}_{h_tag}": 0.0,
        f"clean_retrace_precision_{pair}_{h_tag}": 0.0,
        f"low_squeeze_conversion_{pair}_{h_tag}": 0.0,
        f"pullback_before_squeeze_share_{pair}_{h_tag}": 0.0,
    }


def _build_empty_signal_quality_frame(signals_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = signals_df.copy()
    out['entry_time'] = pd.NaT
    out['entry_price'] = np.nan
    pair = _pair_key(PRIMARY_SQUEEZE_PCT, PRIMARY_PULLBACK_PCT)
    pullback_pct_int = int(round(PRIMARY_PULLBACK_PCT * 100))
    for h in horizons:
        h_tag = f"h{h}"
        out[f"squeeze_pct_{h_tag}"] = np.nan
        out[f"pullback_pct_{h_tag}"] = np.nan
        out[f"net_edge_pct_{h_tag}"] = np.nan
        out[f"prepullback_squeeze_pct_{h_tag}_{pullback_pct_int}"] = np.nan
        out[f"time_to_pullback_{pullback_pct_int}_{h_tag}"] = np.nan
        out[f"time_to_squeeze_{int(round(PRIMARY_SQUEEZE_PCT * 100))}_{h_tag}"] = np.nan
        out[f"clean_{pair}_{h_tag}"] = False
        out[f"dirty_retrace_{pair}_{h_tag}"] = False
        out[f"clean_no_pullback_{pair}_{h_tag}"] = False
        out[f"dirty_no_pullback_{pair}_{h_tag}"] = False
        out[f"pullback_before_squeeze_{pair}_{h_tag}"] = False
    return out


def _resolve_threshold_times_in_1s(
        candles_loader,
        symbol: str,
        minute_start: pd.Timestamp,
        entry_price: float,
        squeeze_threshold: float,
        pullback_threshold: float
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float | None]:
    bars_1s = candles_loader.load_1s_bars_from_transactions(
        symbol,
        minute_start.to_pydatetime(),
        (minute_start + pd.Timedelta(minutes=1)).to_pydatetime()
    )
    if bars_1s.empty:
        return None, None, None

    squeeze_time = None
    pullback_time = None
    prepullback_squeeze = 0.0

    for sec_time, sec_bar in bars_1s.iterrows():
        sec_squeeze = (float(sec_bar['high']) - entry_price) / entry_price
        sec_pullback = (entry_price - float(sec_bar['low'])) / entry_price

        if pullback_time is None:
            if sec_pullback >= pullback_threshold:
                pullback_time = pd.Timestamp(sec_time)
            else:
                prepullback_squeeze = max(prepullback_squeeze, sec_squeeze)

        if squeeze_time is None and sec_squeeze >= squeeze_threshold:
            squeeze_time = pd.Timestamp(sec_time)

        if pullback_time is not None and squeeze_time is not None:
            break

    return squeeze_time, pullback_time, prepullback_squeeze


def _compute_ordered_threshold_metrics(
        window_df: pd.DataFrame,
        entry_price: float,
        entry_time: pd.Timestamp,
        candles_loader,
        symbol: str,
        squeeze_threshold: float,
        pullback_threshold: float
) -> tuple[float, float | None, float | None]:
    squeeze_time = None
    pullback_time = None
    prepullback_squeeze = 0.0

    for minute_time, minute_bar in window_df.iterrows():
        bar_squeeze = (float(minute_bar['high']) - entry_price) / entry_price
        bar_pullback = (entry_price - float(minute_bar['low'])) / entry_price

        same_bar_cross = pullback_time is None and bar_squeeze >= squeeze_threshold and bar_pullback >= pullback_threshold
        if same_bar_cross:
            sec_squeeze_time, sec_pullback_time, sec_prepullback = _resolve_threshold_times_in_1s(
                candles_loader=candles_loader,
                symbol=symbol,
                minute_start=pd.Timestamp(minute_time),
                entry_price=entry_price,
                squeeze_threshold=squeeze_threshold,
                pullback_threshold=pullback_threshold
            )
            if sec_prepullback is not None:
                prepullback_squeeze = max(prepullback_squeeze, sec_prepullback)
            else:
                prepullback_squeeze = max(prepullback_squeeze, bar_squeeze)

            if squeeze_time is None:
                squeeze_time = sec_squeeze_time if sec_squeeze_time is not None else pd.Timestamp(minute_time)
            if pullback_time is None:
                pullback_time = sec_pullback_time if sec_pullback_time is not None else pd.Timestamp(minute_time)
            continue

        if pullback_time is None:
            prepullback_squeeze = max(prepullback_squeeze, bar_squeeze)
            if bar_pullback >= pullback_threshold:
                pullback_time = pd.Timestamp(minute_time)

        if squeeze_time is None and bar_squeeze >= squeeze_threshold:
            squeeze_time = pd.Timestamp(minute_time)

    time_to_pullback_min = None
    time_to_squeeze_min = None
    if pullback_time is not None:
        time_to_pullback_min = (pullback_time - entry_time).total_seconds() / 60.0
    if squeeze_time is not None:
        time_to_squeeze_min = (squeeze_time - entry_time).total_seconds() / 60.0
    return prepullback_squeeze, time_to_pullback_min, time_to_squeeze_min


def _compute_ordered_threshold_metrics_from_arrays(
        window_times: np.ndarray,
        window_high: np.ndarray,
        window_low: np.ndarray,
        entry_price: float,
        entry_time: pd.Timestamp,
        candles_loader,
        symbol: str,
        squeeze_threshold: float,
        pullback_threshold: float
) -> tuple[float, float | None, float | None]:
    squeeze_time = None
    pullback_time = None
    prepullback_squeeze = 0.0

    for i in range(len(window_times)):
        minute_time = pd.Timestamp(window_times[i])
        bar_squeeze = (float(window_high[i]) - entry_price) / entry_price
        bar_pullback = (entry_price - float(window_low[i])) / entry_price

        same_bar_cross = pullback_time is None and bar_squeeze >= squeeze_threshold and bar_pullback >= pullback_threshold
        if same_bar_cross:
            sec_squeeze_time, sec_pullback_time, sec_prepullback = _resolve_threshold_times_in_1s(
                candles_loader=candles_loader,
                symbol=symbol,
                minute_start=minute_time,
                entry_price=entry_price,
                squeeze_threshold=squeeze_threshold,
                pullback_threshold=pullback_threshold
            )
            if sec_prepullback is not None:
                prepullback_squeeze = max(prepullback_squeeze, sec_prepullback)
            else:
                prepullback_squeeze = max(prepullback_squeeze, bar_squeeze)

            if squeeze_time is None:
                squeeze_time = sec_squeeze_time if sec_squeeze_time is not None else minute_time
            if pullback_time is None:
                pullback_time = sec_pullback_time if sec_pullback_time is not None else minute_time
            continue

        if pullback_time is None:
            prepullback_squeeze = max(prepullback_squeeze, bar_squeeze)
            if bar_pullback >= pullback_threshold:
                pullback_time = minute_time

        if squeeze_time is None and bar_squeeze >= squeeze_threshold:
            squeeze_time = minute_time

    time_to_pullback_min = None
    time_to_squeeze_min = None
    if pullback_time is not None:
        time_to_pullback_min = (pullback_time - entry_time).total_seconds() / 60.0
    if squeeze_time is not None:
        time_to_squeeze_min = (squeeze_time - entry_time).total_seconds() / 60.0
    return prepullback_squeeze, time_to_pullback_min, time_to_squeeze_min


def build_signal_path_metrics(
        signals_df: pd.DataFrame,
        candles_loader,
        horizons: list | None = None,
        squeeze_threshold: float = PRIMARY_SQUEEZE_PCT,
        pullback_threshold: float = PRIMARY_PULLBACK_PCT,
        entry_shift_bars: int = 0,
) -> pd.DataFrame:
    if horizons is None:
        horizons = DEFAULT_SIGNAL_QUALITY_HORIZONS
    horizons = sorted(set(int(h) for h in horizons))

    if signals_df.empty:
        return _build_empty_signal_quality_frame(signals_df, horizons)

    signals = signals_df.copy()
    signals['open_time'] = pd.to_datetime(signals['open_time'])
    out = signals.copy()
    out['entry_time'] = pd.NaT
    out['entry_price'] = np.nan

    pair = _pair_key(squeeze_threshold, pullback_threshold)
    squeeze_pct_int = int(round(squeeze_threshold * 100))
    pullback_pct_int = int(round(pullback_threshold * 100))

    one_minute_cache = {}
    max_horizon = max(horizons)
    symbol_ranges = {}
    for symbol, group in signals.groupby('symbol'):
        eval_times = group['open_time'] + pd.Timedelta(minutes=15 * entry_shift_bars)
        start_time = eval_times.min()
        end_time = eval_times.max() + pd.Timedelta(minutes=max_horizon * 15 + 1)
        symbol_ranges[symbol] = (start_time.to_pydatetime(), end_time.to_pydatetime())

    if hasattr(candles_loader, 'load_raw_1m_candles_batch'):
        symbols = list(symbol_ranges.keys())
        global_start = min(v[0] for v in symbol_ranges.values())
        global_end = max(v[1] for v in symbol_ranges.values())
        global_span_minutes = max(1.0, (global_end - global_start).total_seconds() / 60.0)
        range_minutes_total = float(sum(max(0.0, (v[1] - v[0]).total_seconds() / 60.0) for v in symbol_ranges.values()))
        overlap_ratio = range_minutes_total / global_span_minutes if global_span_minutes > 0 else 1.0
        use_batch = len(symbols) >= 20 or overlap_ratio >= 8.0
        if use_batch:
            one_minute_cache = candles_loader.load_raw_1m_candles_batch(
                symbols=symbols,
                start_time=global_start,
                end_time=global_end,
            )
        elif hasattr(candles_loader, 'load_raw_1m_candles_ranges'):
            one_minute_cache = candles_loader.load_raw_1m_candles_ranges(symbol_ranges)
        else:
            one_minute_cache = candles_loader.load_raw_1m_candles_batch(
                symbols=symbols,
                start_time=global_start,
                end_time=global_end,
            )
    elif hasattr(candles_loader, 'load_raw_1m_candles_ranges'):
        one_minute_cache = candles_loader.load_raw_1m_candles_ranges(symbol_ranges)
    else:
        symbols = list(symbol_ranges.keys())
        global_start = min(v[0] for v in symbol_ranges.values())
        global_end = max(v[1] for v in symbol_ranges.values())
        for symbol, (start_time, end_time) in symbol_ranges.items():
            df_1m = candles_loader.load_raw_1m_candles(symbol, start_time, end_time)
            one_minute_cache[symbol] = df_1m

    for h in horizons:
        h_tag = f"h{h}"
        out[f"squeeze_pct_{h_tag}"] = np.nan
        out[f"pullback_pct_{h_tag}"] = np.nan
        out[f"net_edge_pct_{h_tag}"] = np.nan
        out[f"prepullback_squeeze_pct_{h_tag}_{pullback_pct_int}"] = np.nan
        out[f"time_to_pullback_{pullback_pct_int}_{h_tag}"] = np.nan
        out[f"time_to_squeeze_{squeeze_pct_int}_{h_tag}"] = np.nan
        out[f"clean_{pair}_{h_tag}"] = False
        out[f"dirty_retrace_{pair}_{h_tag}"] = False
        out[f"clean_no_pullback_{pair}_{h_tag}"] = False
        out[f"dirty_no_pullback_{pair}_{h_tag}"] = False
        out[f"pullback_before_squeeze_{pair}_{h_tag}"] = False

    n_rows = len(out)
    column_buffers: dict[str, np.ndarray] = {}
    column_buffers['entry_time'] = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    column_buffers['entry_price'] = np.full(n_rows, np.nan, dtype=float)
    for h in horizons:
        h_tag = f"h{h}"
        column_buffers[f"squeeze_pct_{h_tag}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"pullback_pct_{h_tag}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"net_edge_pct_{h_tag}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"prepullback_squeeze_pct_{h_tag}_{pullback_pct_int}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"time_to_pullback_{pullback_pct_int}_{h_tag}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"time_to_squeeze_{squeeze_pct_int}_{h_tag}"] = np.full(n_rows, np.nan, dtype=float)
        column_buffers[f"clean_{pair}_{h_tag}"] = np.zeros(n_rows, dtype=bool)
        column_buffers[f"dirty_retrace_{pair}_{h_tag}"] = np.zeros(n_rows, dtype=bool)
        column_buffers[f"clean_no_pullback_{pair}_{h_tag}"] = np.zeros(n_rows, dtype=bool)
        column_buffers[f"dirty_no_pullback_{pair}_{h_tag}"] = np.zeros(n_rows, dtype=bool)
        column_buffers[f"pullback_before_squeeze_{pair}_{h_tag}"] = np.zeros(n_rows, dtype=bool)

    symbol_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for symbol, df_1m in one_minute_cache.items():
        if df_1m.empty:
            continue
        symbol_cache[str(symbol)] = (
            df_1m.index.to_numpy(dtype="datetime64[ns]"),
            df_1m['open'].to_numpy(dtype=float),
            df_1m['high'].to_numpy(dtype=float),
            df_1m['low'].to_numpy(dtype=float),
        )

    for row_idx, row in enumerate(out.itertuples(index=False)):
        symbol = str(row.symbol)
        signal_time = pd.Timestamp(row.open_time)
        eval_time = signal_time + pd.Timedelta(minutes=15 * entry_shift_bars)
        if symbol not in symbol_cache:
            continue
        idx_values, open_values, high_values, low_values = symbol_cache[symbol]
        eval_np = np.datetime64(eval_time.to_datetime64())
        pos = int(np.searchsorted(idx_values, eval_np, side='left'))
        if pos >= len(idx_values):
            continue
        entry_time = pd.Timestamp(idx_values[pos])
        entry_price = float(open_values[pos])
        if entry_price <= 0:
            continue

        column_buffers['entry_time'][row_idx] = entry_time.to_datetime64()
        column_buffers['entry_price'][row_idx] = entry_price

        for h in horizons:
            h_tag = f"h{h}"
            horizon_end = eval_time + pd.Timedelta(minutes=h * 15)
            end_np = np.datetime64(horizon_end.to_datetime64())
            end_pos = int(np.searchsorted(idx_values, end_np, side='left'))
            if end_pos <= pos:
                continue
            window_times = idx_values[pos:end_pos]
            window_high = high_values[pos:end_pos]
            window_low = low_values[pos:end_pos]

            squeeze_pct = (float(np.max(window_high)) - entry_price) / entry_price
            pullback_pct = (entry_price - float(np.min(window_low))) / entry_price
            net_edge_pct = pullback_pct - squeeze_pct

            prepullback_squeeze, time_to_pullback_min, time_to_squeeze_min = _compute_ordered_threshold_metrics_from_arrays(
                window_times=window_times,
                window_high=window_high,
                window_low=window_low,
                entry_price=entry_price,
                entry_time=eval_time,
                candles_loader=candles_loader,
                symbol=symbol,
                squeeze_threshold=squeeze_threshold,
                pullback_threshold=pullback_threshold
            )

            pullback_hit = pullback_pct >= pullback_threshold
            clean_retrace = pullback_hit and prepullback_squeeze <= squeeze_threshold
            dirty_retrace = pullback_hit and prepullback_squeeze > squeeze_threshold
            clean_no_pullback = (not pullback_hit) and squeeze_pct <= squeeze_threshold
            dirty_no_pullback = (not pullback_hit) and squeeze_pct > squeeze_threshold

            pullback_before_squeeze = False
            if time_to_pullback_min is not None:
                if time_to_squeeze_min is None:
                    pullback_before_squeeze = True
                else:
                    pullback_before_squeeze = time_to_pullback_min < time_to_squeeze_min

            column_buffers[f"squeeze_pct_{h_tag}"][row_idx] = squeeze_pct
            column_buffers[f"pullback_pct_{h_tag}"][row_idx] = pullback_pct
            column_buffers[f"net_edge_pct_{h_tag}"][row_idx] = net_edge_pct
            column_buffers[f"prepullback_squeeze_pct_{h_tag}_{pullback_pct_int}"][row_idx] = prepullback_squeeze
            column_buffers[f"time_to_pullback_{pullback_pct_int}_{h_tag}"][row_idx] = (
                np.nan if time_to_pullback_min is None else float(time_to_pullback_min)
            )
            column_buffers[f"time_to_squeeze_{squeeze_pct_int}_{h_tag}"][row_idx] = (
                np.nan if time_to_squeeze_min is None else float(time_to_squeeze_min)
            )
            column_buffers[f"clean_{pair}_{h_tag}"][row_idx] = bool(clean_retrace)
            column_buffers[f"dirty_retrace_{pair}_{h_tag}"][row_idx] = bool(dirty_retrace)
            column_buffers[f"clean_no_pullback_{pair}_{h_tag}"][row_idx] = bool(clean_no_pullback)
            column_buffers[f"dirty_no_pullback_{pair}_{h_tag}"][row_idx] = bool(dirty_no_pullback)
            column_buffers[f"pullback_before_squeeze_{pair}_{h_tag}"][row_idx] = bool(pullback_before_squeeze)

    for col_name, values in column_buffers.items():
        out[col_name] = values

    return out


def compute_signal_quality_metrics(
        signal_path_df: pd.DataFrame,
        horizon: int = PRIMARY_HORIZON_BARS,
        squeeze_threshold: float = PRIMARY_SQUEEZE_PCT,
        pullback_threshold: float = PRIMARY_PULLBACK_PCT
) -> dict:
    pair = _pair_key(squeeze_threshold, pullback_threshold)
    h_tag = f"h{horizon}"
    squeeze_col = f"squeeze_pct_{h_tag}"
    pullback_col = f"pullback_pct_{h_tag}"
    net_edge_col = f"net_edge_pct_{h_tag}"
    clean_col = f"clean_{pair}_{h_tag}"
    dirty_retrace_col = f"dirty_retrace_{pair}_{h_tag}"
    clean_no_pullback_col = f"clean_no_pullback_{pair}_{h_tag}"
    dirty_no_pullback_col = f"dirty_no_pullback_{pair}_{h_tag}"
    pullback_before_col = f"pullback_before_squeeze_{pair}_{h_tag}"

    if signal_path_df.empty or squeeze_col not in signal_path_df.columns:
        return _empty_signal_quality_metrics(
            horizon=horizon,
            squeeze_threshold=squeeze_threshold,
            pullback_threshold=pullback_threshold
        )

    valid = signal_path_df[signal_path_df[squeeze_col].notna() & signal_path_df[pullback_col].notna()].copy()
    if valid.empty:
        return _empty_signal_quality_metrics(
            horizon=horizon,
            squeeze_threshold=squeeze_threshold,
            pullback_threshold=pullback_threshold
        )

    n = len(valid)
    clean_count = int(valid[clean_col].sum())
    dirty_retrace_count = int(valid[dirty_retrace_col].sum())
    clean_no_pullback_count = int(valid[clean_no_pullback_col].sum())
    dirty_no_pullback_count = int(valid[dirty_no_pullback_col].sum())

    metrics = {
        f"squeeze_median_{h_tag}": float(valid[squeeze_col].median()),
        f"squeeze_mean_{h_tag}": float(valid[squeeze_col].mean()),
        f"squeeze_p75_{h_tag}": float(valid[squeeze_col].quantile(0.75)),
        f"pullback_median_{h_tag}": float(valid[pullback_col].median()),
        f"pullback_mean_{h_tag}": float(valid[pullback_col].mean()),
        f"pullback_p75_{h_tag}": float(valid[pullback_col].quantile(0.75)),
        f"net_edge_median_{h_tag}": float(valid[net_edge_col].median()),
        f"net_edge_mean_{h_tag}": float(valid[net_edge_col].mean()),
        f"clean_{pair}_count_{h_tag}": clean_count,
        f"clean_{pair}_share_{h_tag}": clean_count / n,
        f"dirty_retrace_{pair}_count_{h_tag}": dirty_retrace_count,
        f"dirty_retrace_{pair}_share_{h_tag}": dirty_retrace_count / n,
        f"clean_no_pullback_{pair}_count_{h_tag}": clean_no_pullback_count,
        f"clean_no_pullback_{pair}_share_{h_tag}": clean_no_pullback_count / n,
        f"dirty_no_pullback_{pair}_count_{h_tag}": dirty_no_pullback_count,
        f"dirty_no_pullback_{pair}_share_{h_tag}": dirty_no_pullback_count / n,
        f"clean_to_dirty_failure_ratio_{pair}_{h_tag}": clean_count / max(1, dirty_no_pullback_count),
        f"clean_retrace_precision_{pair}_{h_tag}": clean_count / max(1, clean_count + dirty_retrace_count),
        f"low_squeeze_conversion_{pair}_{h_tag}": clean_count / max(1, clean_count + clean_no_pullback_count),
        f"pullback_before_squeeze_share_{pair}_{h_tag}": float(valid[pullback_before_col].mean()),
    }
    return metrics


def attach_signal_quality_columns(
        signals_df: pd.DataFrame,
        candles_loader,
        horizons: list | None = None,
        squeeze_threshold: float = PRIMARY_SQUEEZE_PCT,
        pullback_threshold: float = PRIMARY_PULLBACK_PCT,
        entry_shift_bars: int = 0,
) -> pd.DataFrame:
    return build_signal_path_metrics(
        signals_df=signals_df,
        candles_loader=candles_loader,
        horizons=horizons,
        squeeze_threshold=squeeze_threshold,
        pullback_threshold=pullback_threshold,
        entry_shift_bars=entry_shift_bars,
    )


def evaluate(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        event_data: dict = None,
        abstain_margin: float = 0.0
) -> dict:
    event_metrics = compute_event_level_metrics(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta,
                                                event_data, abstain_margin=abstain_margin)
    point_metrics = compute_point_level_metrics(predictions_df, threshold)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics
    }


def evaluate_with_trade_quality(
        predictions_df: pd.DataFrame,
        threshold: float,
        candles_loader=None,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        horizons: list = None,
        event_data: dict = None,
        abstain_margin: float = 0.0,
        entry_shift_bars: int = 0,
) -> dict:
    event_metrics = compute_event_level_metrics(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta,
                                                event_data, abstain_margin=abstain_margin)
    point_metrics = compute_point_level_metrics(predictions_df, threshold)

    signals_df = extract_signals(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta,
                                 abstain_margin=abstain_margin)
    trade_metrics = {}
    trade_score = np.nan
    signal_quality = _empty_signal_quality_metrics()

    if candles_loader is not None:
        if horizons is None:
            horizons = DEFAULT_SIGNAL_QUALITY_HORIZONS
        trade_metrics = compute_trade_quality_metrics(signals_df, candles_loader, horizons)
        trade_score = compute_trade_quality_score(trade_metrics)
        signal_path_df = build_signal_path_metrics(
            signals_df=signals_df,
            candles_loader=candles_loader,
            horizons=horizons,
            squeeze_threshold=PRIMARY_SQUEEZE_PCT,
            pullback_threshold=PRIMARY_PULLBACK_PCT,
            entry_shift_bars=entry_shift_bars,
        )
        signal_quality = compute_signal_quality_metrics(
            signal_path_df=signal_path_df,
            horizon=PRIMARY_HORIZON_BARS,
            squeeze_threshold=PRIMARY_SQUEEZE_PCT,
            pullback_threshold=PRIMARY_PULLBACK_PCT
        )

    return {
        'event_level': event_metrics,
        'point_level': point_metrics,
        'trade_quality': trade_metrics,
        'signal_quality': signal_quality,
        'trade_quality_score': trade_score
    }
