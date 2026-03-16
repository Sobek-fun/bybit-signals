from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pump_end_prod.infra.clickhouse import DataLoader


TP_PCT = 4.5
SL_PCT = 10.0
ENTRY_SHIFT_BARS = 1
BAR_MINUTES = 15
STRATEGY_STATE_MODE = 'shadow_all_raw_signals'


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
