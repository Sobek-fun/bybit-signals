from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pump_end_threshold.infra.clickhouse import DataLoader


TP_PCT = 4.5
SL_PCT = 10.0
ENTRY_SHIFT_BARS = 1
BAR_MINUTES = 15


def simulate_trade(loader: DataLoader, symbol: str, signal_time: datetime,
                   tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                   entry_shift_bars: int = ENTRY_SHIFT_BARS,
                   max_horizon_bars: int = 200,
                   candle_cache: dict = None) -> dict:
    entry_time = signal_time + timedelta(minutes=entry_shift_bars * BAR_MINUTES)
    end_time = entry_time + timedelta(minutes=max_horizon_bars * BAR_MINUTES)

    if candle_cache is not None and symbol in candle_cache:
        cached_df, cached_start, cached_end = candle_cache[symbol]
        if entry_time >= cached_start and end_time <= cached_end:
            df = cached_df[(cached_df.index >= entry_time) & (cached_df.index <= end_time)]
        else:
            df = loader.load_raw_1m_candles(symbol, entry_time, end_time)
            candle_cache[symbol] = (df, entry_time, end_time)
    else:
        df = loader.load_raw_1m_candles(symbol, entry_time, end_time)
        if candle_cache is not None:
            candle_cache[symbol] = (df, entry_time, end_time)
    if df.empty:
        return {
            'trade_outcome': 'UNKNOWN',
            'tp_hit': False,
            'sl_hit': False,
            'exit_time': None,
            'entry_price': np.nan,
            'exit_price': np.nan,
            'pnl_pct': np.nan,
            'mfe_pct': np.nan,
            'mae_pct': np.nan,
            'trade_duration_bars': np.nan,
        }

    entry_bar = df.iloc[0]
    entry_price = entry_bar['open']

    if entry_price <= 0:
        return {
            'trade_outcome': 'UNKNOWN',
            'tp_hit': False,
            'sl_hit': False,
            'exit_time': None,
            'entry_price': np.nan,
            'exit_price': np.nan,
            'pnl_pct': np.nan,
            'mfe_pct': np.nan,
            'mae_pct': np.nan,
            'trade_duration_bars': np.nan,
        }

    tp_price = entry_price * (1 - tp_pct / 100)
    sl_price = entry_price * (1 + sl_pct / 100)

    mfe = 0.0
    mae = 0.0

    for i in range(len(df)):
        bar = df.iloc[i]
        bar_low = bar['low']
        bar_high = bar['high']

        excursion_down = (entry_price - bar_low) / entry_price * 100
        excursion_up = (bar_high - entry_price) / entry_price * 100
        mfe = max(mfe, excursion_down)
        mae = max(mae, excursion_up)

        tp_hit = bar_low <= tp_price
        sl_hit = bar_high >= sl_price

        if tp_hit and sl_hit:
            return {
                'trade_outcome': 'AMBIGUOUS',
                'tp_hit': True,
                'sl_hit': True,
                'exit_time': df.index[i],
                'entry_price': entry_price,
                'exit_price': np.nan,
                'pnl_pct': np.nan,
                'mfe_pct': mfe,
                'mae_pct': mae,
                'trade_duration_bars': i,
            }

        if tp_hit:
            return {
                'trade_outcome': 'TP',
                'tp_hit': True,
                'sl_hit': False,
                'exit_time': df.index[i],
                'entry_price': entry_price,
                'exit_price': tp_price,
                'pnl_pct': tp_pct,
                'mfe_pct': mfe,
                'mae_pct': mae,
                'trade_duration_bars': i,
            }

        if sl_hit:
            return {
                'trade_outcome': 'SL',
                'tp_hit': False,
                'sl_hit': True,
                'exit_time': df.index[i],
                'entry_price': entry_price,
                'exit_price': sl_price,
                'pnl_pct': -sl_pct,
                'mfe_pct': mfe,
                'mae_pct': mae,
                'trade_duration_bars': i,
            }

    return {
        'trade_outcome': 'TIMEOUT',
        'tp_hit': False,
        'sl_hit': False,
        'exit_time': df.index[-1] if len(df) > 0 else None,
        'entry_price': entry_price,
        'exit_price': df.iloc[-1]['close'] if len(df) > 0 else np.nan,
        'pnl_pct': (entry_price - df.iloc[-1]['close']) / entry_price * 100 if len(df) > 0 else np.nan,
        'mfe_pct': mfe,
        'mae_pct': mae,
        'trade_duration_bars': len(df),
    }


def build_strategy_state(signals_df: pd.DataFrame, loader: DataLoader,
                         tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                         max_horizon_bars: int = 200) -> pd.DataFrame:
    signals = signals_df.sort_values('open_time').reset_index(drop=True)
    trade_results = []
    candle_cache = {}

    symbol_groups = signals.groupby('symbol')
    for symbol, group in symbol_groups:
        min_time = group['open_time'].min()
        max_time = group['open_time'].max()
        entry_start = min_time + timedelta(minutes=ENTRY_SHIFT_BARS * BAR_MINUTES)
        entry_end = max_time + timedelta(minutes=(ENTRY_SHIFT_BARS + max_horizon_bars) * BAR_MINUTES)

        full_df = loader.load_raw_1m_candles(symbol, entry_start, entry_end)
        if not full_df.empty:
            candle_cache[symbol] = (full_df, entry_start, entry_end)

    for i, sig in signals.iterrows():
        result = simulate_trade(
            loader, sig['symbol'], sig['open_time'],
            tp_pct, sl_pct,
            max_horizon_bars=max_horizon_bars,
            candle_cache=candle_cache
        )
        result['signal_idx'] = i
        result['symbol'] = sig['symbol']
        result['open_time'] = sig['open_time']
        if 'signal_id' in sig.index:
            result['signal_id'] = sig['signal_id']
        else:
            result['signal_id'] = f"{sig['symbol']}|{sig['open_time'].strftime('%Y%m%d_%H%M%S')}"
        trade_results.append(result)

    trades_df = pd.DataFrame(trade_results)
    return trades_df


def compute_strategy_state_at(trades_df: pd.DataFrame, t: datetime,
                              all_signal_times: list[datetime]) -> dict:
    resolved = []
    open_trades = []

    for _, tr in trades_df.iterrows():
        if tr['exit_time'] is not None and tr['exit_time'] <= t and tr['trade_outcome'] in ('TP', 'SL'):
            resolved.append({
                'outcome': tr['trade_outcome'],
                'exit_time': tr['exit_time'],
                'open_time': tr['open_time'],
                'entry_time': tr['open_time'] + timedelta(minutes=ENTRY_SHIFT_BARS * BAR_MINUTES),
            })
        elif tr['open_time'] <= t and (tr['exit_time'] is None or tr['exit_time'] > t):
            open_trades.append({
                'entry_time': tr['open_time'] + timedelta(minutes=ENTRY_SHIFT_BARS * BAR_MINUTES),
                'exit_time': tr.get('exit_time'),
                'open_time': tr['open_time'],
            })

    signals_list = [{'open_time': st} for st in all_signal_times if st <= t]

    return {
        'resolved_trades': resolved,
        'all_signals': signals_list,
        'open_trades': open_trades,
    }


def compute_targets(trades_df: pd.DataFrame, current_idx: int,
                    window_sizes: list[int] = None,
                    min_resolved: int = 3,
                    sl_rate_threshold: float = 0.60) -> dict:
    if window_sizes is None:
        window_sizes = [3, 5]

    targets = {}
    n = len(trades_df)
    current_time = trades_df.iloc[current_idx]['open_time']

    for w in window_sizes:
        end_idx = min(current_idx + 1 + w, n)
        future_slice = trades_df.iloc[current_idx + 1:end_idx]

        resolved = future_slice[future_slice['trade_outcome'].isin(['TP', 'SL'])]
        n_resolved = len(resolved)

        if n_resolved < min_resolved:
            targets[f'target_bad_next_{w}'] = np.nan
            targets[f'target_next_{w}_sl_rate'] = np.nan
            targets[f'target_next_{w}_pnl_sum'] = np.nan
            targets[f'target_future_block_value_{w}'] = np.nan
        else:
            n_sl = int((resolved['trade_outcome'] == 'SL').sum())
            sl_rate = n_sl / n_resolved
            targets[f'target_bad_next_{w}'] = 1 if sl_rate >= sl_rate_threshold else 0
            targets[f'target_next_{w}_sl_rate'] = sl_rate
            targets[f'target_next_{w}_pnl_sum'] = float(resolved['pnl_pct'].sum())

            tp_pnl = (resolved['trade_outcome'] == 'TP').sum() * TP_PCT
            sl_pnl = (resolved['trade_outcome'] == 'SL').sum() * (-SL_PCT)
            targets[f'target_future_block_value_{w}'] = tp_pnl + sl_pnl

        consecutive_sl = 0
        for _, trade in future_slice.iterrows():
            if trade['trade_outcome'] == 'SL':
                consecutive_sl += 1
            elif trade['trade_outcome'] == 'TP':
                break

        targets[f'target_sl_streak_start_{w}'] = 1 if consecutive_sl >= 2 else 0

    future_12h_end = current_time + timedelta(hours=12)
    future_12h = trades_df[
        (trades_df['open_time'] > current_time) &
        (trades_df['open_time'] <= future_12h_end) &
        (trades_df['trade_outcome'].isin(['TP', 'SL', 'TIMEOUT']))
    ]
    n_res_12h = len(future_12h[future_12h['trade_outcome'].isin(['TP', 'SL'])])

    if n_res_12h < min_resolved:
        targets['target_bad_next_12h'] = np.nan
        targets['target_future_pnl_sum_12h'] = np.nan
        targets['target_future_block_value_12h'] = np.nan
        targets['target_drawdown_cluster_next_12h'] = np.nan
    else:
        n_sl_12h = int((future_12h['trade_outcome'] == 'SL').sum())
        n_tp_12h = int((future_12h['trade_outcome'] == 'TP').sum())
        sl_rate_12h = n_sl_12h / n_res_12h
        targets['target_bad_next_12h'] = 1 if sl_rate_12h >= sl_rate_threshold else 0

        pnl_sum_12h = 0.0
        for _, t in future_12h.iterrows():
            if not pd.isna(t['pnl_pct']):
                pnl_sum_12h += t['pnl_pct']
            elif t['trade_outcome'] == 'TP':
                pnl_sum_12h += TP_PCT
            elif t['trade_outcome'] == 'SL':
                pnl_sum_12h -= SL_PCT
        targets['target_future_pnl_sum_12h'] = pnl_sum_12h

        block_value_12h = n_tp_12h * TP_PCT + n_sl_12h * (-SL_PCT)
        targets['target_future_block_value_12h'] = block_value_12h

        cumsum_pnl = 0.0
        max_drawdown_depth = 0.0
        consecutive_sl_count = 0
        max_consecutive_sl = 0

        for _, t in future_12h.iterrows():
            if t['trade_outcome'] == 'SL':
                cumsum_pnl -= SL_PCT
                consecutive_sl_count += 1
                max_consecutive_sl = max(max_consecutive_sl, consecutive_sl_count)
            elif t['trade_outcome'] == 'TP':
                cumsum_pnl += TP_PCT
                consecutive_sl_count = 0

            if cumsum_pnl < max_drawdown_depth:
                max_drawdown_depth = cumsum_pnl

        drawdown_cluster = 1 if (max_drawdown_depth <= -15.0 or max_consecutive_sl >= 3) else 0
        targets['target_drawdown_cluster_next_12h'] = drawdown_cluster

    return targets


def build_regime_dataset(
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        min_resolved: int = 3,
        sl_rate_threshold: float = 0.60,
) -> pd.DataFrame:
    signals = signals_df.sort_values('open_time').reset_index(drop=True)
    trades = trades_df.sort_values('open_time').reset_index(drop=True)

    all_signal_times = signals['open_time'].tolist()

    if 'signal_id' not in signals.columns:
        if 'signal_offset' in signals.columns:
            signals['signal_id'] = [
                f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}|{row.get('signal_offset', 0)}"
                for _, row in signals.iterrows()
            ]
        else:
            signals['signal_id'] = [
                f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
                for _, row in signals.iterrows()
            ]

    if 'signal_id' not in trades.columns:
        if 'signal_offset' in trades.columns:
            trades['signal_id'] = [
                f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}|{row.get('signal_offset', 0)}"
                for _, row in trades.iterrows()
            ]
        else:
            trades['signal_id'] = [
                f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
                for _, row in trades.iterrows()
            ]

    if 'signal_id' not in features_df.columns:
        if 'symbol' in features_df.columns and 'open_time' in features_df.columns:
            if 'signal_offset' in features_df.columns:
                features_df['signal_id'] = [
                    f"{row['symbol']}|{pd.to_datetime(row['open_time']).strftime('%Y%m%d_%H%M%S')}|{row.get('signal_offset', 0)}"
                    for _, row in features_df.iterrows()
                ]
            else:
                features_df['signal_id'] = [
                    f"{row['symbol']}|{pd.to_datetime(row['open_time']).strftime('%Y%m%d_%H%M%S')}"
                    for _, row in features_df.iterrows()
                ]
        else:
            print(f"WARNING: Cannot create signal_id for features_df. Columns: {list(features_df.columns)[:10]}")
            return pd.DataFrame()

    rows = []
    skipped_no_trade = 0
    skipped_no_features = 0

    for i in range(len(signals)):
        sig = signals.iloc[i]
        signal_id = sig['signal_id']

        trade_row = trades[trades['signal_id'] == signal_id]
        if trade_row.empty:
            skipped_no_trade += 1
            continue
        trade = trade_row.iloc[0]

        feature_row = features_df[features_df['signal_id'] == signal_id]
        if feature_row.empty:
            skipped_no_features += 1
            continue
        feats = feature_row.iloc[0].to_dict()

        targets = compute_targets(trades, i, min_resolved=min_resolved, sl_rate_threshold=sl_rate_threshold)

        row = {}
        row.update(feats)
        row['signal_id'] = signal_id
        row['trade_outcome'] = trade['trade_outcome']
        row['tp_hit'] = trade['tp_hit']
        row['sl_hit'] = trade['sl_hit']
        row['exit_time'] = trade['exit_time']
        row['entry_price'] = trade['entry_price']
        row['exit_price'] = trade['exit_price']
        row['pnl_pct'] = trade['pnl_pct']
        row['mfe_pct'] = trade['mfe_pct']
        row['mae_pct'] = trade['mae_pct']
        row['trade_duration_bars'] = trade['trade_duration_bars']
        row.update(targets)

        is_bad_12h = targets.get('target_bad_next_12h', 0)
        pnl_12h = targets.get('target_future_pnl_sum_12h', 0)

        base_weight = 1.0
        if not pd.isna(is_bad_12h) and is_bad_12h == 1:
            severity = abs(pnl_12h) if not pd.isna(pnl_12h) else 0
            if severity >= 20:
                base_weight = 3.0
            elif severity >= 10:
                base_weight = 2.0
            else:
                base_weight = 1.5

        row['sample_weight'] = base_weight

        rows.append(row)

    if skipped_no_trade > 0 or skipped_no_features > 0 or len(rows) == 0:
        print(f"DEBUG: build_regime_dataset - signals: {len(signals)}, trades: {len(trades)}, features: {len(features_df)}")
        print(f"DEBUG: skipped_no_trade: {skipped_no_trade}, skipped_no_features: {skipped_no_features}, rows built: {len(rows)}")
        if len(rows) == 0 and len(signals) > 0:
            print(f"DEBUG: Sample signal_id from signals: {signals.iloc[0]['signal_id']}")
            print(f"DEBUG: Sample signal_id from trades: {trades.iloc[0]['signal_id'] if len(trades) > 0 else 'NO TRADES'}")
            print(f"DEBUG: Sample signal_id from features: {features_df.iloc[0]['signal_id'] if len(features_df) > 0 else 'NO FEATURES'}")

    return pd.DataFrame(rows)
