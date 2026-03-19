from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pump_end_threshold_regime.infra.clickhouse import DataLoader

TP_PCT = 4.5
SL_PCT = 10.0
ENTRY_SHIFT_BARS = 1
BAR_MINUTES = 15
STRATEGY_STATE_MODE = 'shadow_all_raw_signals'

TARGET_PROFILES = {
    'pause_value_12h_v2_all': {
        'window_hours': 12,
        'min_resolved': 4,
        'tp_value': 4.5,
        'sl_value': 10.0,
        'timeout_penalty': 1.5,
        'bad_value_threshold': -10.0,
        'bad_sl_rate_threshold': 0.55,
        'good_value_threshold': 7.5,
        'good_sl_rate_threshold': 0.45,
    },
    'pause_value_12h_v2_curated': {
        'window_hours': 12,
        'min_resolved': 3,
        'tp_value': 4.5,
        'sl_value': 10.0,
        'timeout_penalty': 1.0,
        'bad_value_threshold': -10.0,
        'bad_sl_rate_threshold': 0.55,
        'good_value_threshold': 7.5,
        'good_sl_rate_threshold': 0.45,
    },
    'pause_value_12h_v3_clean_extremes': {
        'window_hours': 12,
        'min_resolved': 2,
        'tp_value': 4.5,
        'sl_value': 10.0,
        'timeout_penalty': 1.0,
        'bad_value_threshold': -5.0,
        'bad_sl_rate_threshold': 0.60,
        'good_value_threshold': 15.0,
        'good_sl_rate_threshold': 0.40,
    },
    'pause_value_12h_v3_loose_bad': {
        'window_hours': 12,
        'min_resolved': 2,
        'tp_value': 4.5,
        'sl_value': 10.0,
        'timeout_penalty': 1.0,
        'bad_value_threshold': -5.0,
        'bad_sl_rate_threshold': 0.50,
        'good_value_threshold': 15.0,
        'good_sl_rate_threshold': 0.40,
    },
}


def simulate_trade_fast_1m(loader: DataLoader, symbol: str, signal_time: datetime,
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
            'ambiguous_bar_time': None,
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
            'ambiguous_bar_time': None,
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
                'ambiguous_bar_time': df.index[i],
                'tp_price': tp_price,
                'sl_price': sl_price,
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
                'ambiguous_bar_time': None,
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
                'ambiguous_bar_time': None,
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
        'ambiguous_bar_time': None,
    }


def resolve_trade_with_1s_replay(loader: DataLoader, symbol: str,
                                 entry_price: float, tp_price: float, sl_price: float,
                                 ambiguous_bar_time: datetime,
                                 mfe: float, mae: float,
                                 trade_duration_bars: int,
                                 tp_pct: float = TP_PCT, sl_pct: float = SL_PCT) -> dict:
    bar_start = ambiguous_bar_time
    bar_end = ambiguous_bar_time + timedelta(minutes=1)

    bars_1s = loader.load_1s_bars_from_transactions(symbol, bar_start, bar_end)

    if bars_1s.empty:
        return {
            'trade_outcome': 'AMBIGUOUS',
            'tp_hit': True,
            'sl_hit': True,
            'exit_time': ambiguous_bar_time,
            'entry_price': entry_price,
            'exit_price': np.nan,
            'pnl_pct': np.nan,
            'mfe_pct': mfe,
            'mae_pct': mae,
            'trade_duration_bars': trade_duration_bars,
        }

    for i in range(len(bars_1s)):
        bar = bars_1s.iloc[i]
        bar_low = bar['low']
        bar_high = bar['high']

        excursion_down = (entry_price - bar_low) / entry_price * 100
        excursion_up = (bar_high - entry_price) / entry_price * 100
        mfe = max(mfe, excursion_down)
        mae = max(mae, excursion_up)

        tp_hit = bar_low <= tp_price
        sl_hit = bar_high >= sl_price

        if tp_hit and not sl_hit:
            return {
                'trade_outcome': 'TP',
                'tp_hit': True,
                'sl_hit': False,
                'exit_time': bars_1s.index[i],
                'entry_price': entry_price,
                'exit_price': tp_price,
                'pnl_pct': tp_pct,
                'mfe_pct': mfe,
                'mae_pct': mae,
                'trade_duration_bars': trade_duration_bars,
            }

        if sl_hit and not tp_hit:
            return {
                'trade_outcome': 'SL',
                'tp_hit': False,
                'sl_hit': True,
                'exit_time': bars_1s.index[i],
                'entry_price': entry_price,
                'exit_price': sl_price,
                'pnl_pct': -sl_pct,
                'mfe_pct': mfe,
                'mae_pct': mae,
                'trade_duration_bars': trade_duration_bars,
            }

    return {
        'trade_outcome': 'AMBIGUOUS',
        'tp_hit': True,
        'sl_hit': True,
        'exit_time': ambiguous_bar_time,
        'entry_price': entry_price,
        'exit_price': np.nan,
        'pnl_pct': np.nan,
        'mfe_pct': mfe,
        'mae_pct': mae,
        'trade_duration_bars': trade_duration_bars,
    }


def simulate_trade(loader: DataLoader, symbol: str, signal_time: datetime,
                   tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                   entry_shift_bars: int = ENTRY_SHIFT_BARS,
                   max_horizon_bars: int = 200,
                   candle_cache: dict = None) -> dict:
    result = simulate_trade_fast_1m(
        loader, symbol, signal_time,
        tp_pct=tp_pct, sl_pct=sl_pct,
        entry_shift_bars=entry_shift_bars,
        max_horizon_bars=max_horizon_bars,
        candle_cache=candle_cache,
    )

    if result['trade_outcome'] == 'AMBIGUOUS' and result.get('ambiguous_bar_time') is not None:
        resolved = resolve_trade_with_1s_replay(
            loader, symbol,
            entry_price=result['entry_price'],
            tp_price=result['tp_price'],
            sl_price=result['sl_price'],
            ambiguous_bar_time=result['ambiguous_bar_time'],
            mfe=result['mfe_pct'],
            mae=result['mae_pct'],
            trade_duration_bars=result['trade_duration_bars'],
            tp_pct=tp_pct, sl_pct=sl_pct,
        )
        return resolved

    clean = {k: v for k, v in result.items() if k not in ('ambiguous_bar_time', 'tp_price', 'sl_price')}
    return clean


def build_strategy_state(signals_df: pd.DataFrame, loader: DataLoader,
                         tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                         max_horizon_bars: int = 200,
                         trade_replay_source: str = "1s") -> pd.DataFrame:
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
        if trade_replay_source == "1m":
            result = simulate_trade_fast_1m(
                loader,
                sig['symbol'],
                sig['open_time'],
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                max_horizon_bars=max_horizon_bars,
                candle_cache=candle_cache
            )
            result = {k: v for k, v in result.items() if k not in ('ambiguous_bar_time', 'tp_price', 'sl_price')}
        else:
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


def simulate_trade_live_asof(loader: DataLoader, symbol: str, signal_time: datetime, asof_time: datetime,
                             tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                             entry_shift_bars: int = ENTRY_SHIFT_BARS,
                             max_horizon_bars: int = 200,
                             trade_replay_source: str = "1s",
                             candle_cache: dict = None) -> dict:
    entry_time = signal_time + timedelta(minutes=entry_shift_bars * BAR_MINUTES)
    horizon_end_time = entry_time + timedelta(minutes=max_horizon_bars * BAR_MINUTES)
    replay_end = min(horizon_end_time, asof_time)

    if replay_end <= entry_time:
        return {
            'trade_outcome': 'OPEN',
            'tp_hit': False,
            'sl_hit': False,
            'exit_time': None,
            'entry_price': np.nan,
            'exit_price': np.nan,
            'pnl_pct': np.nan,
            'mfe_pct': np.nan,
            'mae_pct': np.nan,
            'trade_duration_bars': 0,
        }

    if candle_cache is not None and symbol in candle_cache:
        cached_df, cached_start, cached_end = candle_cache[symbol]
        if entry_time >= cached_start and replay_end <= cached_end:
            df = cached_df[(cached_df.index >= entry_time) & (cached_df.index < replay_end)]
        else:
            df = loader.load_raw_1m_candles(symbol, entry_time, replay_end)
            candle_cache[symbol] = (df, entry_time, replay_end)
    else:
        df = loader.load_raw_1m_candles(symbol, entry_time, replay_end)
        if candle_cache is not None:
            candle_cache[symbol] = (df, entry_time, replay_end)

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
            if trade_replay_source == "1m":
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
            resolved = resolve_trade_with_1s_replay(
                loader,
                symbol,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                ambiguous_bar_time=df.index[i],
                mfe=mfe,
                mae=mae,
                trade_duration_bars=i,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
            )
            return resolved

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

    if asof_time >= horizon_end_time:
        return {
            'trade_outcome': 'TIMEOUT',
            'tp_hit': False,
            'sl_hit': False,
            'exit_time': horizon_end_time,
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['close'] if len(df) > 0 else np.nan,
            'pnl_pct': (entry_price - df.iloc[-1]['close']) / entry_price * 100 if len(df) > 0 else np.nan,
            'mfe_pct': mfe,
            'mae_pct': mae,
            'trade_duration_bars': len(df),
        }

    return {
        'trade_outcome': 'OPEN',
        'tp_hit': False,
        'sl_hit': False,
        'exit_time': None,
        'entry_price': entry_price,
        'exit_price': np.nan,
        'pnl_pct': np.nan,
        'mfe_pct': mfe,
        'mae_pct': mae,
        'trade_duration_bars': len(df),
    }


def build_strategy_state_live(signals_df: pd.DataFrame, loader: DataLoader, asof_time: datetime,
                              tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                              max_horizon_bars: int = 200,
                              trade_replay_source: str = "1s") -> pd.DataFrame:
    signals = signals_df.sort_values('open_time').reset_index(drop=True).copy()
    if signals.empty:
        return pd.DataFrame()

    trade_results = []
    candle_cache = {}

    symbol_groups = signals.groupby('symbol')
    for symbol, group in symbol_groups:
        min_time = group['open_time'].min()
        max_time = group['open_time'].max()
        entry_start = min_time + timedelta(minutes=ENTRY_SHIFT_BARS * BAR_MINUTES)
        entry_end = max_time + timedelta(minutes=(ENTRY_SHIFT_BARS + max_horizon_bars) * BAR_MINUTES)
        replay_end = min(entry_end, asof_time)
        if replay_end > entry_start:
            full_df = loader.load_raw_1m_candles(symbol, entry_start, replay_end)
            if not full_df.empty:
                candle_cache[symbol] = (full_df, entry_start, replay_end)

    for i, sig in signals.iterrows():
        result = simulate_trade_live_asof(
            loader,
            sig['symbol'],
            sig['open_time'],
            asof_time=asof_time,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_horizon_bars=max_horizon_bars,
            trade_replay_source=trade_replay_source,
            candle_cache=candle_cache,
        )
        result['signal_idx'] = i
        result['symbol'] = sig['symbol']
        result['open_time'] = sig['open_time']
        if 'signal_id' in sig.index:
            result['signal_id'] = sig['signal_id']
        else:
            result['signal_id'] = f"{sig['symbol']}|{sig['open_time'].strftime('%Y%m%d_%H%M%S')}"
        trade_results.append(result)

    return pd.DataFrame(trade_results)


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


def compute_pause_value_targets(
        trades_df: pd.DataFrame,
        current_time: datetime,
        window_hours: int = 12,
        min_resolved: int = 4,
        tp_value: float = 4.5,
        sl_value: float = 10.0,
        timeout_penalty: float = 1.5,
        bad_value_threshold: float = -10.0,
        bad_sl_rate_threshold: float = 0.55,
        good_value_threshold: float = 7.5,
        good_sl_rate_threshold: float = 0.45,
) -> dict:
    future_end = current_time + timedelta(hours=window_hours)
    future = trades_df[
        (trades_df['open_time'] > current_time) &
        (trades_df['open_time'] <= future_end)
        ]

    resolved = future[future['trade_outcome'].isin(['TP', 'SL'])]
    timeout = future[future['trade_outcome'] == 'TIMEOUT']

    n_resolved = len(resolved)
    n_tp = int((resolved['trade_outcome'] == 'TP').sum()) if n_resolved > 0 else 0
    n_sl = int((resolved['trade_outcome'] == 'SL').sum()) if n_resolved > 0 else 0
    n_timeout = len(timeout)

    targets = {
        f'future_resolved_count_next_{window_hours}h': n_resolved,
        f'future_tp_count_next_{window_hours}h': n_tp,
        f'future_sl_count_next_{window_hours}h': n_sl,
        f'future_timeout_count_next_{window_hours}h': n_timeout,
    }

    if n_resolved < min_resolved:
        targets[f'future_sl_rate_next_{window_hours}h'] = np.nan
        targets[f'future_block_value_next_{window_hours}h'] = np.nan
        targets[f'target_pause_value_next_{window_hours}h'] = np.nan
        return targets

    sl_rate = n_sl / n_resolved
    block_value = tp_value * n_tp - sl_value * n_sl - timeout_penalty * n_timeout

    targets[f'future_sl_rate_next_{window_hours}h'] = sl_rate
    targets[f'future_block_value_next_{window_hours}h'] = block_value

    if block_value <= bad_value_threshold and sl_rate >= bad_sl_rate_threshold:
        targets[f'target_pause_value_next_{window_hours}h'] = 1
    elif block_value >= good_value_threshold and sl_rate <= good_sl_rate_threshold:
        targets[f'target_pause_value_next_{window_hours}h'] = 0
    else:
        targets[f'target_pause_value_next_{window_hours}h'] = np.nan

    return targets


def compute_targets(trades_df: pd.DataFrame, current_idx: int,
                    window_sizes: list[int] = None,
                    min_resolved: int = 3,
                    sl_rate_threshold: float = 0.60,
                    target_profile: str = None) -> dict:
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

    if target_profile and target_profile in TARGET_PROFILES:
        profile = TARGET_PROFILES[target_profile]
    else:
        profile = TARGET_PROFILES['pause_value_12h_v2_all']

    pause_targets = compute_pause_value_targets(
        trades_df, current_time, **profile
    )
    targets.update(pause_targets)

    return targets


def build_regime_dataset(
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        min_resolved: int = 3,
        sl_rate_threshold: float = 0.60,
        target_profile: str = None,
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

        targets = compute_targets(
            trades, i, min_resolved=min_resolved,
            sl_rate_threshold=sl_rate_threshold,
            target_profile=target_profile,
        )

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

        target_val = targets.get('target_pause_value_next_12h', np.nan)

        if pd.isna(target_val):
            row['sample_weight'] = np.nan
        elif target_val == 1:
            block_val = targets.get('future_block_value_next_12h', 0)
            severity = abs(block_val) if not pd.isna(block_val) else 0
            if severity >= 20:
                row['sample_weight'] = 2.0
            else:
                row['sample_weight'] = 1.5
        else:
            block_val = targets.get('future_block_value_next_12h', 0)
            if not pd.isna(block_val) and block_val >= 15:
                row['sample_weight'] = 1.5
            else:
                row['sample_weight'] = 1.2

        rows.append(row)

    if skipped_no_trade > 0 or skipped_no_features > 0 or len(rows) == 0:
        print(
            f"DEBUG: build_regime_dataset - signals: {len(signals)}, trades: {len(trades)}, features: {len(features_df)}")
        print(
            f"DEBUG: skipped_no_trade: {skipped_no_trade}, skipped_no_features: {skipped_no_features}, rows built: {len(rows)}")
        if len(rows) == 0 and len(signals) > 0:
            print(f"DEBUG: Sample signal_id from signals: {signals.iloc[0]['signal_id']}")
            print(
                f"DEBUG: Sample signal_id from trades: {trades.iloc[0]['signal_id'] if len(trades) > 0 else 'NO TRADES'}")
            print(
                f"DEBUG: Sample signal_id from features: {features_df.iloc[0]['signal_id'] if len(features_df) > 0 else 'NO FEATURES'}")

    return pd.DataFrame(rows)
