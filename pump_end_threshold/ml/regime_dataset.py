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
                   max_horizon_bars: int = 200) -> dict:
    entry_time = signal_time + timedelta(minutes=entry_shift_bars * BAR_MINUTES)
    end_time = entry_time + timedelta(minutes=max_horizon_bars * BAR_MINUTES)

    df = loader.load_candles_range(symbol, entry_time, end_time)
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
                'trade_outcome': 'SL',
                'tp_hit': True,
                'sl_hit': True,
                'exit_time': df.index[i],
                'entry_price': entry_price,
                'exit_price': sl_price,
                'pnl_pct': -sl_pct,
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

    for i, sig in signals.iterrows():
        result = simulate_trade(
            loader, sig['symbol'], sig['open_time'],
            tp_pct, sl_pct,
            max_horizon_bars=max_horizon_bars
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
        n_sl = int((resolved['trade_outcome'] == 'SL').sum())
        sl_rate = n_sl / max(1, n_resolved)

        targets[f'target_bad_next_{w}'] = 1 if (n_resolved >= min_resolved and sl_rate >= sl_rate_threshold) else 0
        targets[f'target_next_{w}_sl_rate'] = sl_rate
        targets[f'target_next_{w}_pnl_sum'] = float(resolved['pnl_pct'].sum()) if n_resolved > 0 else 0.0

    future_12h_end = current_time + timedelta(hours=12)
    future_12h = trades_df[
        (trades_df['open_time'] > current_time) &
        (trades_df['open_time'] <= future_12h_end) &
        (trades_df['trade_outcome'].isin(['TP', 'SL']))
    ]
    n_res_12h = len(future_12h)
    n_sl_12h = int((future_12h['trade_outcome'] == 'SL').sum()) if n_res_12h > 0 else 0
    sl_rate_12h = n_sl_12h / max(1, n_res_12h)
    targets['target_bad_next_12h'] = 1 if (n_res_12h >= min_resolved and sl_rate_12h >= sl_rate_threshold) else 0

    return targets


def build_regime_dataset(
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        trades_df: pd.DataFrame,
) -> pd.DataFrame:
    signals = signals_df.sort_values('open_time').reset_index(drop=True)
    trades = trades_df.sort_values('open_time').reset_index(drop=True)

    all_signal_times = signals['open_time'].tolist()

    if 'signal_id' not in signals.columns:
        signals['signal_id'] = [
            f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
            for _, row in signals.iterrows()
        ]

    if 'signal_id' not in trades.columns:
        trades['signal_id'] = [
            f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
            for _, row in trades.iterrows()
        ]

    if 'signal_id' not in features_df.columns:
        features_df['signal_id'] = [
            f"{row['symbol']}|{row['open_time'].strftime('%Y%m%d_%H%M%S')}"
            for _, row in features_df.iterrows()
        ]

    rows = []
    for i in range(len(signals)):
        sig = signals.iloc[i]
        signal_id = sig['signal_id']

        trade_row = trades[trades['signal_id'] == signal_id]
        if trade_row.empty:
            continue
        trade = trade_row.iloc[0]

        feature_row = features_df[features_df['signal_id'] == signal_id]
        if feature_row.empty:
            continue
        feats = feature_row.iloc[0].to_dict()

        targets = compute_targets(trades, i)

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

        rows.append(row)

    return pd.DataFrame(rows)
