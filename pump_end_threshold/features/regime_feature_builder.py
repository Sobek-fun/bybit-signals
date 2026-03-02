from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta

from pump_end_threshold.infra.clickhouse import DataLoader, get_liquid_universe


class RegimeFeatureBuilder:
    HIGH_8W_BARS = 8 * 7 * 24 * 4

    def __init__(
            self,
            ch_dsn: str,
            liquid_universe: list[str] = None,
            top_n: int = 120,
    ):
        self.ch_dsn = ch_dsn
        self.loader = DataLoader(ch_dsn)
        self.liquid_universe = liquid_universe
        self.top_n = top_n
        self._candle_cache = {}

    def build(
            self,
            signals_df: pd.DataFrame,
            strategy_state: dict = None,
    ) -> pd.DataFrame:
        signals = signals_df.sort_values('open_time').reset_index(drop=True)
        if signals.empty:
            return pd.DataFrame()

        t_min = signals['open_time'].min()
        t_max = signals['open_time'].max()
        warmup = timedelta(hours=8 * 7 * 24)

        btc_candles = self._load_candles_cached('BTCUSDT', t_min - warmup, t_max)
        eth_candles = self._load_candles_cached('ETHUSDT', t_min - warmup, t_max)

        if self.liquid_universe is None:
            self.liquid_universe = get_liquid_universe(
                self.ch_dsn, t_min - timedelta(days=7), t_max, top_n=self.top_n
            )

        breadth_candles = self._load_breadth_candles(t_min - timedelta(hours=96 * 0.25), t_max)

        rows = []
        for _, sig in signals.iterrows():
            row = {}
            ot = sig['open_time']

            row.update(self._detector_confidence_features(sig))

            token_sym = sig['symbol']
            token_candles = self._load_candles_cached(token_sym, t_min - warmup, t_max)
            row.update(self._token_context_features(token_candles, ot))

            row.update(self._asset_context_features(btc_candles, ot, prefix='btc'))
            row.update(self._asset_context_features(eth_candles, ot, prefix='eth'))

            row.update(self._breadth_features(breadth_candles, ot))

            if strategy_state is not None:
                row.update(self._strategy_state_features(strategy_state, ot))

            rows.append(row)

        features_df = pd.DataFrame(rows)

        keep_cols = ['event_id', 'symbol', 'open_time', 'event_type']
        for c in keep_cols:
            if c in signals.columns:
                features_df[c] = signals[c].values

        return features_df

    def _load_candles_cached(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        key = symbol
        if key not in self._candle_cache:
            self._candle_cache[key] = self.loader.load_candles_range(symbol, start, end)
        return self._candle_cache[key]

    def _load_breadth_candles(self, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        if not self.liquid_universe:
            return {}
        batch = self.loader.load_candles_batch(self.liquid_universe, start, end)
        return batch

    def _detector_confidence_features(self, sig: pd.Series) -> dict:
        f = {}
        for col in ['p_end_at_fire', 'threshold_gap', 'pending_bars',
                     'drop_from_peak_at_fire', 'signal_offset']:
            f[f'det_{col}'] = sig.get(col, np.nan)
        return f

    def _token_context_features(self, df: pd.DataFrame, t: datetime) -> dict:
        f = {}
        if df.empty or t not in df.index:
            loc = df.index.searchsorted(t) - 1
            if loc < 0:
                return self._fill_token_context_nan()
        else:
            loc = df.index.get_loc(t)

        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 0 or loc >= len(df):
            return self._fill_token_context_nan()

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        c = close[loc]

        for w in [4, 16, 96]:
            start_idx = max(0, loc - w)
            if start_idx < loc and close[start_idx] > 0:
                f[f'token_ret_{w}'] = (c / close[start_idx]) - 1
            else:
                f[f'token_ret_{w}'] = np.nan

        rolling_max = np.max(high[max(0, loc - 96):loc + 1]) if loc > 0 else c
        f['token_drawdown'] = (c / rolling_max - 1) if rolling_max > 0 else 0.0

        vol_start = max(0, loc - 20)
        vol_slice = volume[vol_start:loc + 1]
        if len(vol_slice) > 1:
            med_vol = np.median(vol_slice[:-1]) if len(vol_slice) > 1 else vol_slice[0]
            f['token_vol_ratio_20'] = vol_slice[-1] / med_vol if med_vol > 0 else np.nan
        else:
            f['token_vol_ratio_20'] = np.nan

        close_slice = close[max(0, loc - 14):loc + 1]
        if len(close_slice) >= 2:
            delta = np.diff(close_slice.astype(float))
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = np.mean(gain) if len(gain) > 0 else 0
            avg_loss = np.mean(loss) if len(loss) > 0 else 1e-9
            rs = avg_gain / max(avg_loss, 1e-9)
            f['token_rsi_14'] = 100 - 100 / (1 + rs)
        else:
            f['token_rsi_14'] = np.nan

        h8w_start = max(0, loc - self.HIGH_8W_BARS)
        high_8w = np.max(high[h8w_start:loc + 1])
        f['token_dist_to_high_8w'] = (c / high_8w - 1) if high_8w > 0 else 0.0
        f['token_within_2pct_high_8w'] = 1 if (high_8w > 0 and c >= high_8w * 0.98) else 0

        high_slice = high[h8w_start:loc + 1]
        if len(high_slice) > 0:
            peak_pos = np.argmax(high_slice)
            f['token_bars_since_high_8w'] = len(high_slice) - 1 - peak_pos
        else:
            f['token_bars_since_high_8w'] = np.nan

        ret1_slice = np.diff(close[max(0, loc - 16):loc + 1].astype(float))
        if len(ret1_slice) > 0:
            streak = 0
            for r in reversed(ret1_slice):
                if r > 0:
                    streak += 1
                else:
                    break
            f['token_up_streak'] = streak
            f['token_green_frac_16'] = np.mean(ret1_slice > 0)
        else:
            f['token_up_streak'] = 0
            f['token_green_frac_16'] = np.nan

        if loc >= 96:
            h96 = np.max(high[loc - 96:loc + 1])
            f['token_near_high_96'] = 1 if (h96 > 0 and c >= h96 * 0.98) else 0
        else:
            f['token_near_high_96'] = np.nan

        return f

    def _fill_token_context_nan(self) -> dict:
        keys = [
            'token_ret_4', 'token_ret_16', 'token_ret_96',
            'token_drawdown', 'token_vol_ratio_20', 'token_rsi_14',
            'token_dist_to_high_8w', 'token_within_2pct_high_8w',
            'token_bars_since_high_8w', 'token_up_streak',
            'token_green_frac_16', 'token_near_high_96',
        ]
        return {k: np.nan for k in keys}

    def _asset_context_features(self, df: pd.DataFrame, t: datetime, prefix: str) -> dict:
        f = {}
        if df.empty:
            return self._fill_asset_nan(prefix)

        if t in df.index:
            loc = df.index.get_loc(t)
        else:
            loc = df.index.searchsorted(t) - 1

        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 0 or loc >= len(df):
            return self._fill_asset_nan(prefix)

        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values
        c = close[loc]

        for w in [16, 96]:
            si = max(0, loc - w)
            if si < loc and close[si] > 0:
                f[f'{prefix}_ret_{w}'] = (c / close[si]) - 1
            else:
                f[f'{prefix}_ret_{w}'] = np.nan

        vol_start = max(0, loc - 20)
        vol_slice = volume[vol_start:loc + 1]
        if len(vol_slice) > 1:
            med_vol = np.median(vol_slice[:-1])
            f[f'{prefix}_vol_ratio_20'] = vol_slice[-1] / med_vol if med_vol > 0 else np.nan
        else:
            f[f'{prefix}_vol_ratio_20'] = np.nan

        ret1_slice = np.diff(close[max(0, loc - 16):loc + 1].astype(float))
        if len(ret1_slice) > 0:
            streak = 0
            for r in reversed(ret1_slice):
                if r > 0:
                    streak += 1
                else:
                    break
            f[f'{prefix}_up_streak'] = streak
            f[f'{prefix}_green_frac_16'] = float(np.mean(ret1_slice > 0))
        else:
            f[f'{prefix}_up_streak'] = 0
            f[f'{prefix}_green_frac_16'] = np.nan

        atr_w = min(14, loc)
        if atr_w >= 1:
            tr = np.maximum(
                high[loc - atr_w:loc + 1] - df['low'].values[loc - atr_w:loc + 1],
                np.maximum(
                    np.abs(high[loc - atr_w:loc + 1] - np.roll(close, 1)[loc - atr_w:loc + 1]),
                    np.abs(df['low'].values[loc - atr_w:loc + 1] - np.roll(close, 1)[loc - atr_w:loc + 1])
                )
            )
            f[f'{prefix}_atr_norm'] = float(np.mean(tr) / c) if c > 0 else np.nan
        else:
            f[f'{prefix}_atr_norm'] = np.nan

        h8w_start = max(0, loc - self.HIGH_8W_BARS)
        high_8w = np.max(high[h8w_start:loc + 1])
        f[f'{prefix}_dist_to_high_8w'] = (c / high_8w - 1) if high_8w > 0 else 0.0
        f[f'{prefix}_within_2pct_high_8w'] = 1 if (high_8w > 0 and c >= high_8w * 0.98) else 0

        high_slice = high[h8w_start:loc + 1]
        if len(high_slice) > 0:
            peak_pos = np.argmax(high_slice)
            f[f'{prefix}_bars_since_high_8w'] = len(high_slice) - 1 - peak_pos
        else:
            f[f'{prefix}_bars_since_high_8w'] = np.nan

        return f

    def _fill_asset_nan(self, prefix: str) -> dict:
        keys = [
            f'{prefix}_ret_16', f'{prefix}_ret_96',
            f'{prefix}_vol_ratio_20', f'{prefix}_up_streak',
            f'{prefix}_green_frac_16', f'{prefix}_atr_norm',
            f'{prefix}_dist_to_high_8w', f'{prefix}_within_2pct_high_8w',
            f'{prefix}_bars_since_high_8w',
        ]
        return {k: np.nan for k in keys}

    def _breadth_features(self, breadth_candles: dict[str, pd.DataFrame], t: datetime) -> dict:
        f = {}
        if not breadth_candles:
            return self._fill_breadth_nan()

        pos_16 = []
        rets_16 = []
        gt_3pct = 0
        gt_5pct = 0
        near_high_96 = 0
        vol_spike_20 = 0
        within_2pct_8w = 0
        within_5pct_8w = 0
        valid_count = 0

        for sym, df in breadth_candles.items():
            if df.empty:
                continue

            if t in df.index:
                loc = df.index.get_loc(t)
            else:
                loc = df.index.searchsorted(t) - 1

            if isinstance(loc, slice):
                loc = loc.stop - 1
            if loc < 0 or loc >= len(df):
                continue

            close = df['close'].values
            high = df['high'].values
            volume = df['volume'].values
            c = close[loc]
            valid_count += 1

            si_16 = max(0, loc - 16)
            if si_16 < loc and close[si_16] > 0:
                ret_16 = (c / close[si_16]) - 1
            else:
                ret_16 = 0.0

            rets_16.append(ret_16)
            if ret_16 > 0:
                pos_16.append(1)
            else:
                pos_16.append(0)

            if ret_16 >= 0.03:
                gt_3pct += 1
            if ret_16 >= 0.05:
                gt_5pct += 1

            if loc >= 96:
                h96 = np.max(high[loc - 96:loc + 1])
                if h96 > 0 and c >= h96 * 0.98:
                    near_high_96 += 1

            vol_start = max(0, loc - 20)
            vol_slice = volume[vol_start:loc + 1]
            if len(vol_slice) > 1:
                med_vol = np.median(vol_slice[:-1])
                if med_vol > 0 and vol_slice[-1] >= med_vol * 5.0:
                    vol_spike_20 += 1

            h8w_start = max(0, loc - self.HIGH_8W_BARS)
            high_8w = np.max(high[h8w_start:loc + 1])
            if high_8w > 0:
                if c >= high_8w * 0.98:
                    within_2pct_8w += 1
                if c >= high_8w * 0.95:
                    within_5pct_8w += 1

        if valid_count == 0:
            return self._fill_breadth_nan()

        n = valid_count
        f['breadth_pos_16'] = sum(pos_16) / n
        f['breadth_mean_ret_16'] = float(np.mean(rets_16))
        f['breadth_median_ret_16'] = float(np.median(rets_16))
        f['breadth_share_gt_3pct_16'] = gt_3pct / n
        f['breadth_share_gt_5pct_16'] = gt_5pct / n
        f['breadth_share_near_high_96'] = near_high_96 / n
        f['breadth_share_vol_spike_20'] = vol_spike_20 / n
        f['breadth_share_within_2pct_high_8w'] = within_2pct_8w / n
        f['breadth_share_within_5pct_high_8w'] = within_5pct_8w / n

        return f

    def _fill_breadth_nan(self) -> dict:
        keys = [
            'breadth_pos_16', 'breadth_mean_ret_16', 'breadth_median_ret_16',
            'breadth_share_gt_3pct_16', 'breadth_share_gt_5pct_16',
            'breadth_share_near_high_96', 'breadth_share_vol_spike_20',
            'breadth_share_within_2pct_high_8w', 'breadth_share_within_5pct_high_8w',
        ]
        return {k: np.nan for k in keys}

    def _strategy_state_features(self, state: dict, t: datetime) -> dict:
        resolved = state.get('resolved_trades', [])
        all_signals = state.get('all_signals', [])
        open_trades = state.get('open_trades', [])

        resolved_before = [tr for tr in resolved if tr['exit_time'] <= t]
        signals_before = [s for s in all_signals if s['open_time'] <= t]
        open_now = [tr for tr in open_trades if tr['entry_time'] <= t and tr.get('exit_time', t + timedelta(hours=1)) > t]

        f = {}

        t_1h = t - timedelta(hours=1)
        t_4h = t - timedelta(hours=4)
        t_12h = t - timedelta(hours=12)
        t_24h = t - timedelta(hours=24)

        f['raw_signals_last_1h'] = sum(1 for s in signals_before if s['open_time'] > t_1h)
        f['raw_signals_last_4h'] = sum(1 for s in signals_before if s['open_time'] > t_4h)
        f['raw_signals_last_12h'] = sum(1 for s in signals_before if s['open_time'] > t_12h)
        f['open_trades_now'] = len(open_now)

        resolved_24h = [tr for tr in resolved_before if tr['exit_time'] > t_24h]
        f['resolved_trades_last_24h'] = len(resolved_24h)

        outcomes_24h = [tr['outcome'] for tr in resolved_24h]
        sl_24h = sum(1 for o in outcomes_24h if o == 'SL')
        f['resolved_sl_rate_last_24h'] = sl_24h / max(1, len(outcomes_24h))
        f['resolved_pnl_sum_last_24h'] = sum(
            4.5 if tr['outcome'] == 'TP' else -10.0
            for tr in resolved_24h
        )

        last_5 = resolved_before[-5:] if len(resolved_before) >= 5 else resolved_before
        if last_5:
            sl_5 = sum(1 for tr in last_5 if tr['outcome'] == 'SL')
            f['resolved_sl_rate_last_5'] = sl_5 / len(last_5)
        else:
            f['resolved_sl_rate_last_5'] = np.nan

        tp_streak = 0
        sl_streak = 0
        for tr in reversed(resolved_before):
            if tr['outcome'] == 'TP':
                if sl_streak == 0:
                    tp_streak += 1
                else:
                    break
            elif tr['outcome'] == 'SL':
                if tp_streak == 0:
                    sl_streak += 1
                else:
                    break
            else:
                break

        f['prev_closed_tp_streak'] = tp_streak
        f['prev_closed_sl_streak'] = sl_streak

        return f
