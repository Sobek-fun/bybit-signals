import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)

from pump_end.ml.threshold import _prepare_event_data, _compute_event_metrics_from_data


def compute_event_level_metrics(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1,
        event_data: dict = None
) -> dict:
    if event_data is None:
        event_data = _prepare_event_data(predictions_df)

    hit0 = 0
    hit1 = 0
    early = 0
    late = 0
    miss = 0
    offsets = []
    early_offsets = []

    for event_id, data in event_data.items():
        if signal_rule == 'first_cross':
            mask = data['p_end'] >= threshold
            if not mask.any():
                miss += 1
                continue
            first_idx = np.argmax(mask)
            offset = data['offsets'][first_idx]
        elif signal_rule == 'argmax_per_event':
            argmax_idx = np.argmax(data['p_end'])
            offset = data['offsets'][argmax_idx]
        else:
            offsets_arr = data['offsets']
            p_end = data['p_end']

            triggered = False
            pending_count = 0
            pending_max = -np.inf
            turn_down_count = 0

            for i in range(len(offsets_arr)):
                if p_end[i] >= threshold:
                    pending_count += 1
                    pending_max = max(pending_max, p_end[i])

                    if i > 0 and p_end[i] < p_end[i - 1]:
                        turn_down_count += 1
                    else:
                        turn_down_count = 0

                    if pending_count >= min_pending_bars and pending_max >= min_pending_peak and i > 0:
                        drop_from_peak = pending_max - p_end[i]
                        if drop_from_peak >= drop_delta and turn_down_count >= min_turn_down_bars:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0
                    pending_max = -np.inf
                    turn_down_count = 0

            if not triggered:
                miss += 1
                continue

        offsets.append(offset)

        if offset == 0:
            hit0 += 1
        elif offset == 1:
            hit1 += 1
        elif offset < 0:
            early += 1
            early_offsets.append(offset)
        else:
            late += 1

    n_events = len(event_data)

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
        'avg_pred_offset': np.mean(offsets) if offsets else None,
        'median_pred_offset': np.median(offsets) if offsets else None
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


def compute_offset_distribution(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1
) -> dict:
    event_data = _prepare_event_data(predictions_df)

    offsets = []

    for event_id, data in event_data.items():
        if signal_rule == 'first_cross':
            mask = data['p_end'] >= threshold
            if not mask.any():
                continue
            first_idx = np.argmax(mask)
            offset = data['offsets'][first_idx]
        elif signal_rule == 'argmax_per_event':
            argmax_idx = np.argmax(data['p_end'])
            offset = data['offsets'][argmax_idx]
        else:
            offsets_arr = data['offsets']
            p_end = data['p_end']

            triggered = False
            pending_count = 0
            pending_max = -np.inf
            turn_down_count = 0

            for i in range(len(offsets_arr)):
                if p_end[i] >= threshold:
                    pending_count += 1
                    pending_max = max(pending_max, p_end[i])

                    if i > 0 and p_end[i] < p_end[i - 1]:
                        turn_down_count += 1
                    else:
                        turn_down_count = 0

                    if pending_count >= min_pending_bars and pending_max >= min_pending_peak and i > 0:
                        drop_from_peak = pending_max - p_end[i]
                        if drop_from_peak >= drop_delta and turn_down_count >= min_turn_down_bars:
                            offset = offsets_arr[i]
                            triggered = True
                            break
                else:
                    pending_count = 0
                    pending_max = -np.inf
                    turn_down_count = 0

            if not triggered:
                continue

        offsets.append(offset)

    if not offsets:
        return {
            'offset_minus2_to_plus2_rate': 0.0,
            'tail_le_minus10_rate': 0.0,
            'mean_offset': None,
            'median_offset': None,
            'total_signals': 0
        }

    offsets_arr = np.array(offsets)
    n_signals = len(offsets_arr)

    in_range = np.sum((offsets_arr >= -2) & (offsets_arr <= 2))
    tail_early = np.sum(offsets_arr <= -10)

    return {
        'offset_minus2_to_plus2_rate': in_range / n_signals,
        'tail_le_minus10_rate': tail_early / n_signals,
        'mean_offset': float(np.mean(offsets_arr)),
        'median_offset': float(np.median(offsets_arr)),
        'total_signals': n_signals
    }


def evaluate(
        predictions_df: pd.DataFrame,
        threshold: float,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1,
        event_data: dict = None
) -> dict:
    event_metrics = compute_event_level_metrics(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta,
                                                min_pending_peak, min_turn_down_bars, event_data)
    point_metrics = compute_point_level_metrics(predictions_df, threshold)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics
    }


def evaluate_with_trade_quality(
        predictions_df: pd.DataFrame,
        threshold: float,
        candles_loader,
        signal_rule: str = 'pending_turn_down',
        min_pending_bars: int = 1,
        drop_delta: float = 0.0,
        min_pending_peak: float = 0.0,
        min_turn_down_bars: int = 1,
        horizons: list = None,
        event_data: dict = None
) -> dict:
    from pump_end.ml.predict import extract_signals

    event_metrics = compute_event_level_metrics(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta,
                                                min_pending_peak, min_turn_down_bars, event_data)
    point_metrics = compute_point_level_metrics(predictions_df, threshold)

    signals_df = extract_signals(predictions_df, threshold, signal_rule, min_pending_bars, drop_delta, min_pending_peak, min_turn_down_bars)
    trade_metrics = compute_trade_quality_metrics(signals_df, candles_loader, horizons)
    trade_score = compute_trade_quality_score(trade_metrics)

    return {
        'event_level': event_metrics,
        'point_level': point_metrics,
        'trade_quality': trade_metrics,
        'trade_quality_score': trade_score
    }
