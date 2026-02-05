import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from pump_long.threshold import _prepare_event_data


def compute_event_level_metrics_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross',
        event_data: dict = None
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    hit0 = 0
    hitM1 = 0
    early = 0
    late = 0
    miss = 0
    pre_window = 0
    offsets = []

    for event_id, data in event_data.items():
        p_arr = data['p_long']
        offsets_arr = data['offsets']

        if len(p_arr) == 0:
            miss += 1
            continue

        if p_arr[0] >= threshold:
            pre_window += 1
            continue

        cross_idx = None
        for i in range(1, len(p_arr)):
            prev_p = p_arr[i - 1]
            curr_p = p_arr[i]
            if prev_p < threshold <= curr_p:
                cross_idx = i
                break

        if cross_idx is None:
            miss += 1
            continue

        offset = offsets_arr[cross_idx]
        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == -1:
            hitM1 += 1
        elif offset < -1:
            early += 1
        else:
            late += 1

    n_events = len(event_data)

    return {
        'n_events': n_events,
        'hit0': hit0,
        'hit0_rate': hit0 / n_events if n_events > 0 else 0,
        'hitM1': hitM1,
        'hitM1_rate': hitM1 / n_events if n_events > 0 else 0,
        'hit0_or_hitM1': hit0 + hitM1,
        'hit0_or_hitM1_rate': (hit0 + hitM1) / n_events if n_events > 0 else 0,
        'early': early,
        'early_rate': early / n_events if n_events > 0 else 0,
        'late': late,
        'late_rate': late / n_events if n_events > 0 else 0,
        'miss': miss,
        'miss_rate': miss / n_events if n_events > 0 else 0,
        'pre_window': pre_window,
        'pre_window_rate': pre_window / n_events if n_events > 0 else 0,
        'avg_pred_offset': np.mean(offsets) if offsets else None,
        'median_pred_offset': np.median(offsets) if offsets else None
    }


def compute_point_level_metrics_long(
        predictions_df: pd.DataFrame,
        threshold: float
) -> dict:
    y_true = predictions_df['y'].values
    y_prob = predictions_df['p_long'].values
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


def evaluate_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross',
        event_data: dict = None
) -> dict:
    event_metrics = compute_event_level_metrics_long(predictions_df, threshold, signal_rule, event_data)
    point_metrics = compute_point_level_metrics_long(predictions_df, threshold)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics
    }


def extract_signals_event_windows_long(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'first_cross'
) -> pd.DataFrame:
    event_data = _prepare_event_data(predictions_df)

    symbol_map = predictions_df.groupby('event_id')['symbol'].first().to_dict()
    time_map = {}
    for event_id, group in predictions_df.groupby('event_id'):
        sorted_group = group.sort_values('offset')
        time_map[event_id] = dict(zip(sorted_group['offset'], sorted_group['open_time']))

    signals = []

    for event_id, data in event_data.items():
        p_arr = data['p_long']
        offsets_arr = data['offsets']

        if len(p_arr) == 0:
            continue

        if p_arr[0] >= threshold:
            continue

        cross_idx = None
        for i in range(1, len(p_arr)):
            prev_p = p_arr[i - 1]
            curr_p = p_arr[i]
            if prev_p < threshold <= curr_p:
                cross_idx = i
                break

        if cross_idx is None:
            continue

        offset = offsets_arr[cross_idx]

        signals.append({
            'symbol': symbol_map[event_id],
            'open_time': time_map[event_id][offset]
        })

    return pd.DataFrame(signals)


def extract_signals_prodlike_long(
        stream_df: pd.DataFrame,
        threshold: float,
        cooldown_bars: int = 8
) -> pd.DataFrame:
    signals = []

    for symbol, group in stream_df.groupby('symbol'):
        group = group.sort_values('open_time').reset_index(drop=True)
        p_long = group['p_long'].values
        open_times = group['open_time'].values

        n = len(group)
        last_signal_idx = -cooldown_bars - 1

        for i in range(1, n):
            prev_p = p_long[i - 1]
            curr_p = p_long[i]

            if np.isnan(prev_p) or np.isnan(curr_p):
                continue

            is_cross_up = prev_p < threshold <= curr_p
            cooldown_ok = (i - last_signal_idx) > cooldown_bars

            if is_cross_up and cooldown_ok:
                signals.append({
                    'open_time': open_times[i],
                    'symbol': symbol,
                    'p_long': curr_p,
                    'prev_p_long': prev_p,
                    'threshold': threshold,
                    'rule': 'cross_up',
                    'cooldown_bars': cooldown_bars
                })
                last_signal_idx = i

    if not signals:
        return pd.DataFrame(columns=['open_time', 'symbol', 'p_long', 'prev_p_long', 'threshold', 'rule', 'cooldown_bars'])

    return pd.DataFrame(signals).sort_values(['open_time', 'symbol']).reset_index(drop=True)


def match_signals_to_events(
        signals_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        window_before: int = 60,
        window_after: int = 10
) -> pd.DataFrame:
    if signals_df.empty:
        return signals_df.assign(
            matched_event_id=None,
            matched_event_time=pd.NaT,
            offset_to_event=np.nan
        )

    signals_df = signals_df.copy()
    signals_df['matched_event_id'] = None
    signals_df['matched_event_time'] = pd.NaT
    signals_df['offset_to_event'] = np.nan

    if 'event_time' in labels_df.columns:
        event_time_series = pd.to_datetime(labels_df['event_time'])
    elif 'event_open_time' in labels_df.columns:
        event_time_series = pd.to_datetime(labels_df['event_open_time'])
    else:
        event_time_series = pd.to_datetime(labels_df['open_time'])

    event_id_series = labels_df['symbol'] + '|' + event_time_series.dt.strftime('%Y%m%d_%H%M%S')

    bar_seconds = 15 * 60

    symbol_events_map = {}
    for symbol in labels_df['symbol'].unique():
        mask = labels_df['symbol'] == symbol
        group_event_times = event_time_series[mask]
        group_event_ids = event_id_series[mask]
        sort_idx = group_event_times.argsort()
        sorted_times = group_event_times.iloc[sort_idx]
        sorted_ids = group_event_ids.iloc[sort_idx]
        symbol_events_map[symbol] = {
            'times_int': sorted_times.values.astype('datetime64[s]').astype(np.int64),
            'event_ids': sorted_ids.values,
            'event_times': sorted_times.values
        }

    signal_times = pd.to_datetime(signals_df['open_time']).values.astype('datetime64[s]').astype(np.int64)
    signal_symbols = signals_df['symbol'].values

    matched_event_ids = np.empty(len(signals_df), dtype=object)
    matched_event_times = np.empty(len(signals_df), dtype='datetime64[ns]')
    matched_offsets = np.full(len(signals_df), np.nan)

    matched_event_ids[:] = None
    matched_event_times[:] = np.datetime64('NaT')

    for i in range(len(signals_df)):
        symbol = signal_symbols[i]
        if symbol not in symbol_events_map:
            continue

        events_data = symbol_events_map[symbol]
        event_times_int = events_data['times_int']
        event_ids = events_data['event_ids']
        event_times_dt = events_data['event_times']

        signal_time_int = signal_times[i]

        idx = np.searchsorted(event_times_int, signal_time_int)

        candidates = []
        for check_idx in [idx - 1, idx]:
            if 0 <= check_idx < len(event_times_int):
                event_time_int = event_times_int[check_idx]
                offset_seconds = signal_time_int - event_time_int
                offset_bars = int(offset_seconds / bar_seconds)

                if -window_before <= offset_bars <= window_after:
                    candidates.append((check_idx, offset_bars))

        if not candidates:
            continue

        best_idx = None
        best_offset = None
        best_abs_offset = float('inf')

        for check_idx, offset_bars in candidates:
            abs_offset = abs(offset_bars)
            is_better = False
            if abs_offset < best_abs_offset:
                is_better = True
            elif abs_offset == best_abs_offset and offset_bars <= 0 and (best_offset is None or best_offset > 0):
                is_better = True

            if is_better:
                best_idx = check_idx
                best_offset = offset_bars
                best_abs_offset = abs_offset

        if best_idx is not None:
            matched_event_ids[i] = event_ids[best_idx]
            matched_event_times[i] = event_times_dt[best_idx]
            matched_offsets[i] = best_offset

    signals_df['matched_event_id'] = matched_event_ids
    signals_df['matched_event_time'] = matched_event_times
    signals_df['offset_to_event'] = matched_offsets

    return signals_df


def compute_stream_metrics_long(
        signals_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        holdout_start: pd.Timestamp,
        holdout_end: pd.Timestamp,
        window_before: int = 60,
        window_after: int = 10
) -> dict:
    matched_signals = match_signals_to_events(signals_df, labels_df, window_before, window_after)

    total_signals = len(matched_signals)
    tp_signals = matched_signals['matched_event_id'].notna().sum()
    fp_signals = total_signals - tp_signals

    if 'event_open_time' in labels_df.columns:
        event_times = pd.to_datetime(labels_df['event_open_time'])
    else:
        event_times = pd.to_datetime(labels_df['open_time'])

    holdout_mask = (event_times >= holdout_start) & (event_times < holdout_end)
    holdout_events = labels_df[holdout_mask]
    total_events = len(holdout_events)

    caught_event_ids = matched_signals['matched_event_id'].dropna().unique()
    caught_events = len(caught_event_ids)

    holdout_seconds = (holdout_end - holdout_start).total_seconds()
    holdout_days = holdout_seconds / 86400
    if holdout_days <= 0:
        holdout_days = 1

    signals_per_day = total_signals / holdout_days
    precision = tp_signals / total_signals if total_signals > 0 else 0
    event_recall = caught_events / total_events if total_events > 0 else 0

    offsets = matched_signals['offset_to_event'].dropna().values
    offset_distribution = {}
    if len(offsets) > 0:
        offset_distribution = {
            'mean': float(np.mean(offsets)),
            'median': float(np.median(offsets)),
            'std': float(np.std(offsets)),
            'min': float(np.min(offsets)),
            'max': float(np.max(offsets))
        }

    unique_symbols_signaled = 0
    avg_signals_per_symbol = 0.0
    tp_symbols = 0
    fp_symbols = 0

    if total_signals > 0 and 'symbol' in matched_signals.columns:
        unique_symbols_signaled = matched_signals['symbol'].nunique()
        avg_signals_per_symbol = total_signals / unique_symbols_signaled if unique_symbols_signaled > 0 else 0

        tp_mask = matched_signals['matched_event_id'].notna()
        tp_symbols = matched_signals[tp_mask]['symbol'].nunique()

        fp_mask = matched_signals['matched_event_id'].isna()
        fp_symbols = matched_signals[fp_mask]['symbol'].nunique()

    return {
        'total_signals': total_signals,
        'tp_signals': int(tp_signals),
        'fp_signals': int(fp_signals),
        'total_events': total_events,
        'caught_events': caught_events,
        'missed_events': total_events - caught_events,
        'signals_per_day': round(signals_per_day, 2),
        'precision': round(precision, 4),
        'event_recall': round(event_recall, 4),
        'holdout_days': holdout_days,
        'offset_distribution': offset_distribution,
        'unique_symbols_signaled': unique_symbols_signaled,
        'avg_signals_per_symbol': round(avg_signals_per_symbol, 2),
        'tp_symbols': tp_symbols,
        'fp_symbols': fp_symbols
    }
