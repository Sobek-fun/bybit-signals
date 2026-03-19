from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pump_end_threshold_regime.infra.clickhouse import DataLoader, get_liquid_universe


class RegimeFeatureBuilder:

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
        self._breadth_cache = None
        self._breadth_cache_range = None
        self._1m_candle_cache = {}
        self._1s_bar_cache = {}

    def build(
            self,
            signals_df: pd.DataFrame,
            trades_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        signals = signals_df.sort_values('open_time').reset_index(drop=True)
        if signals.empty:
            return pd.DataFrame()

        t_min = signals['open_time'].min()
        t_max = signals['open_time'].max()
        warmup = timedelta(hours=48)

        btc_candles = self._load_candles_cached('BTCUSDT', t_min - warmup, t_max)
        eth_candles = self._load_candles_cached('ETHUSDT', t_min - warmup, t_max)

        if self.liquid_universe is None:
            self.liquid_universe = get_liquid_universe(
                self.ch_dsn, t_min - timedelta(days=7), t_min, top_n=self.top_n
            )

        breadth_lookback = timedelta(hours=48)
        breadth_candles = self._load_breadth_candles(t_min - breadth_lookback, t_max)

        self._preload_1m_candles('BTCUSDT', t_min - timedelta(minutes=20), t_max)
        self._preload_1m_candles('ETHUSDT', t_min - timedelta(minutes=20), t_max)

        rows = []
        for i, sig in signals.iterrows():
            row = {}
            ot = sig['open_time']

            token_sym = sig['symbol']
            token_candles = self._load_candles_cached(token_sym, t_min - warmup, t_max)
            token_features = self._token_context_features(token_candles, ot)
            row.update(token_features)

            btc_features = self._asset_context_features(btc_candles, ot, prefix='btc')
            eth_features = self._asset_context_features(eth_candles, ot, prefix='eth')
            row.update(btc_features)
            row.update(eth_features)

            breadth_features = self._breadth_features(breadth_candles, ot)
            row.update(breadth_features)

            token_relative_features = self._token_relative_context_features(
                token_features, btc_features, eth_features, breadth_features
            )
            row.update(token_relative_features)

            row.update(self._market_interaction_features(btc_features, eth_features))

            signals_history = signals[signals['open_time'] < ot]
            signals_bucket = signals[signals['open_time'] == ot]
            row.update(self._signal_flow_features(signals_history, sig))
            row['bucket_signals_now'] = len(signals_bucket)

            if trades_df is not None:
                row.update(self._strategy_state_features(trades_df, ot, signals_history, sig['symbol']))

            row.update(self._microstructure_features('BTCUSDT', ot, 'btc'))
            row.update(self._microstructure_features('ETHUSDT', ot, 'eth'))
            self._preload_1m_candles(token_sym, ot - timedelta(minutes=20), ot)
            row.update(self._microstructure_features(token_sym, ot, 'token'))

            btc_loc = btc_candles.index.searchsorted(ot) - 1
            if 0 <= btc_loc < len(btc_candles):
                row['context_time_used'] = btc_candles.index[btc_loc]
            else:
                row['context_time_used'] = pd.NaT

            rows.append(row)

        features_df = pd.DataFrame(rows)

        keep_cols = ['event_id', 'symbol', 'open_time', 'event_type', 'signal_id', 'signal_offset']
        for c in keep_cols:
            if c in signals.columns:
                features_df[c] = signals[c].values

        return features_df

    def _load_candles_cached(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        key = (symbol, start.isoformat(), end.isoformat())
        if key not in self._candle_cache:
            if symbol in self._candle_cache:
                existing_df = self._candle_cache[symbol]
                existing_start = existing_df.index.min() if not existing_df.empty else None
                existing_end = existing_df.index.max() if not existing_df.empty else None

                if existing_start and existing_end:
                    if start >= existing_start and end <= existing_end:
                        return existing_df[(existing_df.index >= start) & (existing_df.index <= end)]

                    if start < existing_start:
                        df_before = self.loader.load_candles_range(symbol, start, existing_start)
                        existing_df = pd.concat([df_before, existing_df]).sort_index()
                        existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
                    if end > existing_end:
                        df_after = self.loader.load_candles_range(symbol, existing_end, end)
                        existing_df = pd.concat([existing_df, df_after]).sort_index()
                        existing_df = existing_df[~existing_df.index.duplicated(keep='first')]

                    self._candle_cache[symbol] = existing_df
                    return existing_df[(existing_df.index >= start) & (existing_df.index <= end)]

            df = self.loader.load_candles_range(symbol, start, end)
            self._candle_cache[symbol] = df
            self._candle_cache[key] = df
            return df
        return self._candle_cache[key]

    def _load_breadth_candles(self, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        if not self.liquid_universe:
            return {}

        if self._breadth_cache is not None and self._breadth_cache_range is not None:
            cache_start, cache_end = self._breadth_cache_range
            if start >= cache_start and end <= cache_end:
                return self._breadth_cache

        self._breadth_cache = self.loader.load_candles_batch(self.liquid_universe, start, end)
        self._breadth_cache_range = (start, end)
        return self._breadth_cache

    def _preload_1m_candles(self, symbol: str, start: datetime, end: datetime):
        if symbol in self._1m_candle_cache:
            cached_df = self._1m_candle_cache[symbol]
            if not cached_df.empty:
                existing_start = cached_df.index.min()
                existing_end = cached_df.index.max()
                if start >= existing_start and end <= existing_end:
                    return
        df = self.loader.load_raw_1m_candles(symbol, start, end)
        if symbol in self._1m_candle_cache and not self._1m_candle_cache[symbol].empty:
            combined = pd.concat([self._1m_candle_cache[symbol], df]).sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            self._1m_candle_cache[symbol] = combined
        else:
            self._1m_candle_cache[symbol] = df

    def _get_1m_candles(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if symbol in self._1m_candle_cache:
            cached = self._1m_candle_cache[symbol]
            if not cached.empty:
                return cached[(cached.index >= start) & (cached.index <= end)]
        df = self.loader.load_raw_1m_candles(symbol, start, end)
        return df

    def _get_1s_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        key = (symbol, start.isoformat(), end.isoformat())
        if key in self._1s_bar_cache:
            return self._1s_bar_cache[key]
        df = self.loader.load_1s_bars_from_transactions(symbol, start, end)
        self._1s_bar_cache[key] = df
        return df

    def _token_context_features(self, df: pd.DataFrame, t: datetime) -> dict:
        f = {}
        if df.empty:
            return self._fill_token_context_nan()
        loc = df.index.searchsorted(t) - 1
        if loc < 0:
            return self._fill_token_context_nan()

        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 0 or loc >= len(df):
            return self._fill_token_context_nan()

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        c = close[loc]

        for w in [1, 4, 16, 96]:
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

        return f

    def _fill_token_context_nan(self) -> dict:
        keys = [
            'token_ret_1', 'token_ret_4', 'token_ret_16', 'token_ret_96',
            'token_drawdown', 'token_vol_ratio_20', 'token_rsi_14',
            'token_up_streak', 'token_green_frac_16',
        ]
        return {k: np.nan for k in keys}

    def _asset_context_features(self, df: pd.DataFrame, t: datetime, prefix: str) -> dict:
        f = {}
        if df.empty:
            return self._fill_asset_nan(prefix)

        loc = df.index.searchsorted(t) - 1

        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 0 or loc >= len(df):
            return self._fill_asset_nan(prefix)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        c = close[loc]

        for w in [1, 4, 16, 96]:
            si = max(0, loc - w)
            if si < loc and close[si] > 0:
                f[f'{prefix}_ret_{w}'] = (c / close[si]) - 1
            else:
                f[f'{prefix}_ret_{w}'] = np.nan

        if loc >= 4:
            f[f'{prefix}_ret_accel'] = f[f'{prefix}_ret_1'] - f[f'{prefix}_ret_4']
        else:
            f[f'{prefix}_ret_accel'] = np.nan

        if loc >= 16:
            f[f'{prefix}_ret_accel_slow'] = f[f'{prefix}_ret_4'] - f[f'{prefix}_ret_16']
        else:
            f[f'{prefix}_ret_accel_slow'] = np.nan

        vol_start = max(0, loc - 20)
        vol_slice = volume[vol_start:loc + 1]
        if len(vol_slice) > 1:
            med_vol = np.median(vol_slice[:-1])
            f[f'{prefix}_vol_ratio_20'] = vol_slice[-1] / med_vol if med_vol > 0 else np.nan
        else:
            f[f'{prefix}_vol_ratio_20'] = np.nan

        ret1_slice = np.diff(close[max(0, loc - 16):loc + 1].astype(float))
        if len(ret1_slice) > 0:
            f[f'{prefix}_green_frac_16'] = float(np.mean(ret1_slice > 0))
        else:
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

        if loc > 0:
            h = high[loc]
            l = low[loc]
            o = df['open'].values[loc] if 'open' in df.columns else c
            bar_range = h - l
            f[f'{prefix}_close_pos'] = (c - l) / bar_range if bar_range > 0 else 0.5
            f[f'{prefix}_upper_wick_ratio'] = (h - max(o, c)) / bar_range if bar_range > 0 else 0.0
            f[f'{prefix}_lower_wick_ratio'] = (min(o, c) - l) / bar_range if bar_range > 0 else 0.0
        else:
            f[f'{prefix}_close_pos'] = 0.5
            f[f'{prefix}_upper_wick_ratio'] = 0.0
            f[f'{prefix}_lower_wick_ratio'] = 0.0

        if loc >= 16:
            h16 = np.max(high[loc - 16:loc + 1])
            f[f'{prefix}_dist_to_high_4h'] = (c / h16 - 1) if h16 > 0 else 0.0
        else:
            f[f'{prefix}_dist_to_high_4h'] = 0.0

        if loc >= 96:
            h96 = np.max(high[loc - 96:loc + 1])
            f[f'{prefix}_dist_to_high_24h'] = (c / h96 - 1) if h96 > 0 else 0.0
        else:
            f[f'{prefix}_dist_to_high_24h'] = 0.0

        if loc >= 16:
            pullback = 0
            for i in range(max(0, loc - 16), loc):
                if close[i] < close[max(0, i - 1)] * 0.995:
                    pullback = 1
                    break
            f[f'{prefix}_pullback_last_4h'] = pullback
        else:
            f[f'{prefix}_pullback_last_4h'] = 0

        return f

    def _fill_asset_nan(self, prefix: str) -> dict:
        keys = [
            f'{prefix}_ret_1', f'{prefix}_ret_4', f'{prefix}_ret_16', f'{prefix}_ret_96',
            f'{prefix}_ret_accel', f'{prefix}_ret_accel_slow',
            f'{prefix}_vol_ratio_20',
            f'{prefix}_green_frac_16', f'{prefix}_atr_norm',
            f'{prefix}_close_pos', f'{prefix}_upper_wick_ratio', f'{prefix}_lower_wick_ratio',
            f'{prefix}_dist_to_high_4h',
            f'{prefix}_dist_to_high_24h', f'{prefix}_pullback_last_4h',
        ]
        return {k: np.nan for k in keys}

    def _breadth_features(self, breadth_candles: dict[str, pd.DataFrame], t: datetime) -> dict:
        f = {}
        if not breadth_candles:
            return self._fill_breadth_nan()

        pos_1 = []
        pos_4 = []
        pos_16 = []
        pos_96 = []
        rets_1 = []
        rets_4 = []
        rets_16 = []
        rets_96 = []
        gt_3pct = 0
        gt_5pct = 0
        near_high_96 = 0
        vol_spike_20 = 0
        valid_count = 0

        for sym, df in breadth_candles.items():
            if df.empty:
                continue

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

            for w, rets_list, pos_list in [(1, rets_1, pos_1), (4, rets_4, pos_4),
                                           (16, rets_16, pos_16), (96, rets_96, pos_96)]:
                si = max(0, loc - w)
                if si < loc and close[si] > 0:
                    ret = (c / close[si]) - 1
                else:
                    ret = 0.0
                rets_list.append(ret)
                pos_list.append(1 if ret > 0 else 0)

            ret_16 = rets_16[-1] if rets_16 else 0.0
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

        if valid_count == 0:
            return self._fill_breadth_nan()

        n = valid_count

        for w, rets_list, pos_list in [(1, rets_1, pos_1), (4, rets_4, pos_4),
                                       (16, rets_16, pos_16), (96, rets_96, pos_96)]:
            f[f'breadth_pos_{w}'] = sum(pos_list) / n
            f[f'breadth_mean_ret_{w}'] = float(np.mean(rets_list))
            f[f'breadth_median_ret_{w}'] = float(np.median(rets_list))

        f['breadth_share_gt_3pct_16'] = gt_3pct / n
        f['breadth_share_gt_5pct_16'] = gt_5pct / n
        f['breadth_share_near_high_96'] = near_high_96 / n
        f['breadth_share_vol_spike_20'] = vol_spike_20 / n
        f['breadth_change_1_to_4'] = f['breadth_pos_1'] - f['breadth_pos_4']
        f['breadth_change_4_to_16'] = f['breadth_pos_4'] - f['breadth_pos_16']
        f['breadth_change_16_to_96'] = f['breadth_pos_16'] - f['breadth_pos_96']

        f['breadth_dispersion_16'] = float(np.std(rets_16))
        f['breadth_acceleration'] = f['breadth_mean_ret_1'] - f['breadth_mean_ret_4']

        f['breadth_dispersion_1'] = float(np.std(rets_1)) if rets_1 else np.nan
        f['breadth_dispersion_4'] = float(np.std(rets_4)) if rets_4 else np.nan
        f['breadth_acceleration_4_to_16'] = f['breadth_mean_ret_4'] - f['breadth_mean_ret_16']

        new_high_1h = 0
        new_high_4h = 0
        new_high_24h = 0
        strong_green_1 = 0
        strong_green_4 = 0

        for sym, df in breadth_candles.items():
            if df.empty:
                continue
            loc = df.index.searchsorted(t) - 1
            if isinstance(loc, slice):
                loc = loc.stop - 1
            if loc < 0 or loc >= len(df):
                continue

            close = df['close'].values
            high = df['high'].values
            c = close[loc]

            if loc >= 4:
                h1 = np.max(high[loc - 4:loc + 1])
                if c >= h1 * 0.99:
                    new_high_1h += 1

            if loc >= 16:
                h4 = np.max(high[loc - 16:loc + 1])
                if c >= h4 * 0.99:
                    new_high_4h += 1

            if loc >= 96:
                h24 = np.max(high[loc - 96:loc + 1])
                if c >= h24 * 0.99:
                    new_high_24h += 1

            if loc >= 1:
                ret_1 = (c / close[loc - 1]) - 1
                if ret_1 > 0.02:
                    strong_green_1 += 1

            if loc >= 4:
                ret_4 = (c / close[loc - 4]) - 1
                if ret_4 > 0.04:
                    strong_green_4 += 1

        f['breadth_new_high_1h'] = new_high_1h / n
        f['breadth_new_high_4h'] = new_high_4h / n
        f['breadth_new_high_24h'] = new_high_24h / n
        f['breadth_strong_green_1'] = strong_green_1 / n
        f['breadth_strong_green_4'] = strong_green_4 / n

        if rets_16:
            rets_16_sorted = sorted(rets_16, reverse=True)
            top_10_idx = max(1, len(rets_16_sorted) // 10)
            top_decile_mean = np.mean(rets_16_sorted[:top_10_idx])
            bottom_decile_mean = np.mean(rets_16_sorted[-top_10_idx:])
            f['breadth_top_decile_ret_16'] = float(top_decile_mean)
            f['breadth_bottom_decile_ret_16'] = float(bottom_decile_mean)

            absolute_rets = [abs(r) for r in rets_16_sorted]
            total_abs = sum(absolute_rets)
            if total_abs > 0:
                cum_share = 0
                for i, abs_ret in enumerate(sorted(absolute_rets, reverse=True)):
                    cum_share += abs_ret / total_abs
                    if cum_share >= 0.5:
                        f['breadth_concentration_50pct'] = (i + 1) / len(absolute_rets)
                        break
                else:
                    f['breadth_concentration_50pct'] = 1.0
            else:
                f['breadth_concentration_50pct'] = 1.0
        else:
            f['breadth_top_decile_ret_16'] = np.nan
            f['breadth_bottom_decile_ret_16'] = np.nan
            f['breadth_concentration_50pct'] = np.nan

        return f

    def _fill_breadth_nan(self) -> dict:
        keys = []
        for w in [1, 4, 16, 96]:
            keys.extend([
                f'breadth_pos_{w}',
                f'breadth_mean_ret_{w}',
                f'breadth_median_ret_{w}',
            ])
        keys.extend([
            'breadth_share_gt_3pct_16', 'breadth_share_gt_5pct_16',
            'breadth_share_near_high_96', 'breadth_share_vol_spike_20',
            'breadth_change_1_to_4', 'breadth_change_4_to_16', 'breadth_change_16_to_96',
            'breadth_dispersion_16', 'breadth_acceleration',
            'breadth_dispersion_1', 'breadth_dispersion_4',
            'breadth_acceleration_4_to_16',
            'breadth_new_high_1h', 'breadth_new_high_4h', 'breadth_new_high_24h',
            'breadth_strong_green_1', 'breadth_strong_green_4',
            'breadth_top_decile_ret_16', 'breadth_bottom_decile_ret_16',
            'breadth_concentration_50pct',
        ])
        return {k: np.nan for k in keys}

    def _token_relative_context_features(self, token_features: dict, btc_features: dict,
                                         eth_features: dict, breadth_features: dict) -> dict:
        f = {}

        def get_safe(d, key, default=np.nan):
            return d.get(key, default) if d else default

        token_ret_1 = get_safe(token_features, 'token_ret_1', 0.0)
        token_ret_4 = get_safe(token_features, 'token_ret_4', 0.0)
        token_ret_16 = get_safe(token_features, 'token_ret_16', 0.0)
        btc_ret_1 = get_safe(btc_features, 'btc_ret_1', 0.0)
        btc_ret_4 = get_safe(btc_features, 'btc_ret_4', 0.0)
        btc_ret_16 = get_safe(btc_features, 'btc_ret_16', 0.0)
        eth_ret_1 = get_safe(eth_features, 'eth_ret_1', 0.0)
        eth_ret_4 = get_safe(eth_features, 'eth_ret_4', 0.0)
        eth_ret_16 = get_safe(eth_features, 'eth_ret_16', 0.0)

        if not pd.isna(btc_ret_16):
            f['token_vs_btc_ret_1'] = token_ret_1 - btc_ret_1
            f['token_vs_btc_ret_4'] = token_ret_4 - btc_ret_4
            f['token_vs_btc_ret_16'] = token_ret_16 - btc_ret_16
            f['token_vs_btc_acceleration'] = (token_ret_1 - btc_ret_1) - (token_ret_4 - btc_ret_4)

        if not pd.isna(eth_ret_16):
            f['token_vs_eth_ret_1'] = token_ret_1 - eth_ret_1
            f['token_vs_eth_ret_4'] = token_ret_4 - eth_ret_4
            f['token_vs_eth_ret_16'] = token_ret_16 - eth_ret_16
            f['token_vs_eth_acceleration'] = (token_ret_1 - eth_ret_1) - (token_ret_4 - eth_ret_4)

        breadth_mean_16 = get_safe(breadth_features, 'breadth_mean_ret_16', 0.0)
        breadth_median_16 = get_safe(breadth_features, 'breadth_median_ret_16', 0.0)

        if not pd.isna(breadth_mean_16):
            f['token_vs_breadth_mean_16'] = token_ret_16 - breadth_mean_16
            f['token_vs_breadth_median_16'] = token_ret_16 - breadth_median_16

        token_vol_ratio = get_safe(token_features, 'token_vol_ratio_20', 1.0)
        breadth_vol_spike_share = get_safe(breadth_features, 'breadth_share_vol_spike_20', 0.0)

        if not pd.isna(token_vol_ratio) and not pd.isna(breadth_vol_spike_share):
            f['token_vol_spike_relative'] = 1 if (token_vol_ratio > 5 and breadth_vol_spike_share < 0.2) else 0

        return f

    def _market_interaction_features(self, btc_features: dict, eth_features: dict) -> dict:
        f = {}

        def get_safe(d, key, default=np.nan):
            return d.get(key, default) if d else default

        btc_ret_16 = get_safe(btc_features, 'btc_ret_16', 0.0)
        eth_ret_16 = get_safe(eth_features, 'eth_ret_16', 0.0)

        if not pd.isna(btc_ret_16) and not pd.isna(eth_ret_16):
            f['btc_eth_divergence'] = btc_ret_16 - eth_ret_16

        return f

    def _signal_flow_features(self, signals_before: pd.DataFrame, current_signal: pd.Series) -> dict:
        t = current_signal['open_time']

        f = {}

        t_1h = t - timedelta(hours=1)
        t_4h = t - timedelta(hours=4)
        t_12h = t - timedelta(hours=12)
        t_24h = t - timedelta(hours=24)

        signals_1h = signals_before[signals_before['open_time'] > t_1h]
        signals_4h = signals_before[signals_before['open_time'] > t_4h]
        signals_12h = signals_before[signals_before['open_time'] > t_12h]
        signals_24h = signals_before[signals_before['open_time'] > t_24h]

        f['raw_signals_last_1h'] = len(signals_1h)
        f['raw_signals_last_4h'] = len(signals_4h)
        f['raw_signals_last_12h'] = len(signals_12h)
        f['raw_signals_last_24h'] = len(signals_24h)

        if len(signals_4h) > 0:
            symbol_counts = signals_4h['symbol'].value_counts()
            f['unique_symbols_last_4h'] = len(symbol_counts)
            f['max_symbol_concentration_4h'] = symbol_counts.max() / len(signals_4h) if len(signals_4h) > 0 else 0.0
        else:
            f['unique_symbols_last_4h'] = 0
            f['max_symbol_concentration_4h'] = 0.0

        if len(signals_24h) > 0:
            f['unique_symbols_last_24h'] = signals_24h['symbol'].nunique()
        else:
            f['unique_symbols_last_24h'] = 0

        for w_hours, w_label in [(4, '4h'), (24, '24h')]:
            w_time = t - timedelta(hours=w_hours)
            w_signals = signals_before[signals_before['open_time'] > w_time]
            if len(w_signals) > 1:
                times = w_signals['open_time'].sort_values()
                gaps = [(times.iloc[i] - times.iloc[i - 1]).total_seconds() / 60 for i in range(1, len(times))]
                f[f'signal_density_median_gap_{w_label}'] = float(np.median(gaps))
            else:
                f[f'signal_density_median_gap_{w_label}'] = np.nan

        return f

    def _strategy_state_features(self, trades_df: pd.DataFrame, t: datetime,
                                 signals_before: pd.DataFrame = None,
                                 current_symbol: str = None) -> dict:
        f = {}

        resolved_before = trades_df[
            (trades_df['exit_time'].notna()) &
            (trades_df['exit_time'] < t) &
            (trades_df['trade_outcome'].isin(['TP', 'SL']))
            ].sort_values('exit_time')

        t_24h = t - timedelta(hours=24)
        resolved_24h = resolved_before[resolved_before['exit_time'] > t_24h]

        if len(resolved_24h) > 0:
            n_sl = (resolved_24h['trade_outcome'] == 'SL').sum()
            f['strat_resolved_sl_rate_last_24h'] = n_sl / len(resolved_24h)
            f['strat_resolved_pnl_sum_last_24h'] = float(resolved_24h['pnl_pct'].fillna(0).sum())
        else:
            f['strat_resolved_sl_rate_last_24h'] = np.nan
            f['strat_resolved_pnl_sum_last_24h'] = np.nan

        f['strat_closed_last_24h'] = len(resolved_24h)

        sl_streak = 0
        tp_streak = 0
        last_closed_is_sl = np.nan
        if len(resolved_before) > 0:
            outcomes = resolved_before['trade_outcome'].tolist()
            for outcome in reversed(outcomes):
                if outcome == 'SL':
                    sl_streak += 1
                else:
                    break
            for outcome in reversed(outcomes):
                if outcome == 'TP':
                    tp_streak += 1
                else:
                    break
            last_closed_is_sl = 1 if outcomes[-1] == 'SL' else 0

        f['strat_prev_closed_sl_streak'] = sl_streak
        f['strat_prev_closed_tp_streak'] = tp_streak
        f['strat_last_closed_is_sl'] = last_closed_is_sl

        last_5 = resolved_before.tail(5)
        if len(last_5) > 0:
            n_sl_5 = (last_5['trade_outcome'] == 'SL').sum()
            n_tp_5 = (last_5['trade_outcome'] == 'TP').sum()
            f['strat_resolved_sl_rate_last_5'] = n_sl_5 / len(last_5)
            f['strat_resolved_tp_rate_last_5'] = n_tp_5 / len(last_5)
            f['strat_resolved_pnl_sum_last_5'] = float(last_5['pnl_pct'].fillna(0).sum())
        else:
            f['strat_resolved_sl_rate_last_5'] = np.nan
            f['strat_resolved_tp_rate_last_5'] = np.nan
            f['strat_resolved_pnl_sum_last_5'] = np.nan

        open_trades = trades_df[
            (trades_df['open_time'] < t) &
            ((trades_df['exit_time'].isna()) | (trades_df['exit_time'] >= t))
            ]
        f['strat_open_trades_now'] = len(open_trades)

        if signals_before is not None:
            t_1h = t - timedelta(hours=1)
            t_4h = t - timedelta(hours=4)
            t_12h = t - timedelta(hours=12)
            f['strat_signals_last_1h'] = len(signals_before[signals_before['open_time'] > t_1h])
            f['strat_signals_last_4h'] = len(signals_before[signals_before['open_time'] > t_4h])
            f['strat_signals_last_12h'] = len(signals_before[signals_before['open_time'] > t_12h])

            signals_4h = signals_before[signals_before['open_time'] > t_4h]
            f['strat_unique_symbols_last_4h'] = signals_4h['symbol'].nunique() if len(signals_4h) > 0 else 0
        else:
            f['strat_signals_last_1h'] = np.nan
            f['strat_signals_last_4h'] = np.nan
            f['strat_signals_last_12h'] = np.nan
            f['strat_unique_symbols_last_4h'] = np.nan

        return f

    def _microstructure_features(self, symbol: str, t: datetime, prefix: str) -> dict:
        f = {}
        start = t - timedelta(minutes=20)
        end = t - timedelta(microseconds=1)

        bars_1m = self._get_1m_candles(symbol, start, end)
        if bars_1m.empty or len(bars_1m) < 2:
            return self._fill_microstructure_nan(prefix)

        loc = bars_1m.index.searchsorted(t) - 1
        if loc < 0 or loc >= len(bars_1m):
            return self._fill_microstructure_nan(prefix)

        close = bars_1m['close'].values
        high = bars_1m['high'].values
        low = bars_1m['low'].values
        c = close[loc]

        if c <= 0:
            return self._fill_microstructure_nan(prefix)

        if loc >= 1 and close[loc - 1] > 0:
            f[f'{prefix}_ret_1m'] = (c / close[loc - 1]) - 1
        else:
            f[f'{prefix}_ret_1m'] = np.nan

        if loc >= 5 and close[loc - 5] > 0:
            f[f'{prefix}_ret_5m'] = (c / close[loc - 5]) - 1
        else:
            f[f'{prefix}_ret_5m'] = np.nan

        if loc >= 15 and close[loc - 15] > 0:
            f[f'{prefix}_ret_15m'] = (c / close[loc - 15]) - 1
        else:
            f[f'{prefix}_ret_15m'] = np.nan

        bar_range = high[loc] - low[loc]
        f[f'{prefix}_range_norm_1m'] = bar_range / c if c > 0 else np.nan

        if loc >= 4:
            h5 = np.max(high[loc - 4:loc + 1])
            l5 = np.min(low[loc - 4:loc + 1])
            f[f'{prefix}_range_norm_5m'] = (h5 - l5) / c if c > 0 else np.nan
        else:
            f[f'{prefix}_range_norm_5m'] = np.nan

        for w, label in [(1, '1m'), (5, '5m'), (15, '15m')]:
            if loc >= w and close[loc - w] > 0:
                price_change = abs(c - close[loc - w])
                h_range = np.max(high[max(0, loc - w):loc + 1]) - np.min(low[max(0, loc - w):loc + 1])
                f[f'{prefix}_trend_eff_{label}'] = price_change / h_range if h_range > 0 else np.nan
            else:
                f[f'{prefix}_trend_eff_{label}'] = np.nan

        for w, label in [(5, '5m'), (15, '15m')]:
            if loc >= w:
                low_w = np.min(low[max(0, loc - w):loc + 1])
                f[f'{prefix}_dist_from_low_{label}'] = (c / low_w - 1) if low_w > 0 else np.nan
            else:
                f[f'{prefix}_dist_from_low_{label}'] = np.nan

        if loc >= 5:
            log_close = np.log(close[loc - 5:loc + 1].astype(float))
            log_rets = np.diff(log_close)
            f[f'{prefix}_rv_5m'] = float(np.std(log_rets)) if len(log_rets) > 1 else np.nan
        else:
            f[f'{prefix}_rv_5m'] = np.nan

        f[f'{prefix}_rv_1m'] = bar_range / c if c > 0 else np.nan

        try:
            tx_start = t - timedelta(minutes=5)
            tx_end = t
            tx_df = self._get_1s_bars(symbol, tx_start, tx_end)
            if not tx_df.empty:
                tx_1m = tx_df[
                    (tx_df.index >= t - timedelta(minutes=1)) &
                    (tx_df.index < t)
                    ]
                tx_5m = tx_df[
                    (tx_df.index >= t - timedelta(minutes=5)) &
                    (tx_df.index < t)
                    ]
                f[f'{prefix}_trades_count_1m'] = int(tx_1m['trades_count'].sum()) if not tx_1m.empty else 0
                f[f'{prefix}_trades_count_5m'] = int(tx_5m['trades_count'].sum()) if not tx_5m.empty else 0
            else:
                f[f'{prefix}_trades_count_1m'] = np.nan
                f[f'{prefix}_trades_count_5m'] = np.nan
        except Exception:
            f[f'{prefix}_trades_count_1m'] = np.nan
            f[f'{prefix}_trades_count_5m'] = np.nan

        return f

    def _fill_microstructure_nan(self, prefix: str) -> dict:
        keys = [
            f'{prefix}_ret_1m', f'{prefix}_ret_5m', f'{prefix}_ret_15m',
            f'{prefix}_range_norm_1m', f'{prefix}_range_norm_5m',
            f'{prefix}_trend_eff_1m', f'{prefix}_trend_eff_5m', f'{prefix}_trend_eff_15m',
            f'{prefix}_dist_from_low_5m', f'{prefix}_dist_from_low_15m',
            f'{prefix}_rv_1m', f'{prefix}_rv_5m',
            f'{prefix}_trades_count_1m', f'{prefix}_trades_count_5m',
        ]
        return {k: np.nan for k in keys}

    def _signal_flow_features_batch(self, all_signals: pd.DataFrame, signal_indices: list) -> pd.DataFrame:
        all_signals_sorted = all_signals.sort_values('open_time').reset_index(drop=True)

        features = []
        for idx in signal_indices:
            sig = all_signals_sorted.iloc[idx]
            ot = sig['open_time']
            signals_history = all_signals_sorted[all_signals_sorted['open_time'] < ot]
            f = self._signal_flow_features(signals_history, sig)
            f['signal_id'] = sig.get('signal_id', idx)
            features.append(f)

        return pd.DataFrame(features)

    def build_batch(self, signals: pd.DataFrame, batch_size: int = 100, trades_df: pd.DataFrame = None) -> pd.DataFrame:
        if signals.empty:
            return pd.DataFrame()

        signals_with_idx = signals.reset_index(drop=True).copy()
        signals_with_idx['_orig_idx'] = range(len(signals_with_idx))
        signals_sorted = signals_with_idx.sort_values('open_time').reset_index(drop=True)

        t_min = signals_sorted['open_time'].min()
        t_max = signals_sorted['open_time'].max()
        warmup = timedelta(hours=48)

        btc_candles = self._load_candles_cached('BTCUSDT', t_min - warmup, t_max)
        eth_candles = self._load_candles_cached('ETHUSDT', t_min - warmup, t_max)

        if self.liquid_universe is None:
            self.liquid_universe = get_liquid_universe(
                self.ch_dsn, t_min - timedelta(days=7), t_min, top_n=self.top_n
            )

        breadth_lookback = timedelta(hours=48)
        breadth_candles = self._load_breadth_candles(t_min - breadth_lookback, t_max)

        self._preload_1m_candles('BTCUSDT', t_min - timedelta(minutes=20), t_max)
        self._preload_1m_candles('ETHUSDT', t_min - timedelta(minutes=20), t_max)

        unique_symbols = signals_sorted['symbol'].unique()
        token_candles_cache = {}
        for sym in unique_symbols:
            token_candles_cache[sym] = self._load_candles_cached(sym, t_min - warmup, t_max)
            self._preload_1m_candles(sym, t_min - timedelta(minutes=20), t_max)

        all_features = []

        for batch_start in range(0, len(signals_sorted), batch_size):
            batch_end = min(batch_start + batch_size, len(signals_sorted))

            batch_features = []
            for i in range(batch_start, batch_end):
                sig = signals_sorted.iloc[i]

                row = {}
                ot = sig['open_time']

                row['_orig_idx'] = sig['_orig_idx']

                token_sym = sig['symbol']
                token_candles = token_candles_cache[token_sym]
                token_features = self._token_context_features(token_candles, ot)
                row.update(token_features)

                btc_features = self._asset_context_features(btc_candles, ot, prefix='btc')
                eth_features = self._asset_context_features(eth_candles, ot, prefix='eth')
                row.update(btc_features)
                row.update(eth_features)

                breadth_features = self._breadth_features(breadth_candles, ot)
                row.update(breadth_features)

                token_relative_features = self._token_relative_context_features(
                    token_features, btc_features, eth_features, breadth_features
                )
                row.update(token_relative_features)

                row.update(self._market_interaction_features(btc_features, eth_features))

                signals_history = signals_sorted[signals_sorted['open_time'] < ot]
                signals_bucket = signals_sorted[signals_sorted['open_time'] == ot]
                row.update(self._signal_flow_features(signals_history, sig))
                row['bucket_signals_now'] = len(signals_bucket)

                if trades_df is not None:
                    row.update(self._strategy_state_features(trades_df, ot, signals_history, sig['symbol']))

                row.update(self._microstructure_features('BTCUSDT', ot, 'btc'))
                row.update(self._microstructure_features('ETHUSDT', ot, 'eth'))
                row.update(self._microstructure_features(token_sym, ot, 'token'))

                btc_loc = btc_candles.index.searchsorted(ot) - 1
                if 0 <= btc_loc < len(btc_candles):
                    row['context_time_used'] = btc_candles.index[btc_loc]
                else:
                    row['context_time_used'] = pd.NaT

                batch_features.append(row)

            all_features.extend(batch_features)

        features_df = pd.DataFrame(all_features)

        features_df = features_df.sort_values('_orig_idx').reset_index(drop=True)
        features_df = features_df.drop('_orig_idx', axis=1)

        if 'symbol' in signals.columns:
            features_df['symbol'] = signals['symbol'].values
        if 'open_time' in signals.columns:
            features_df['open_time'] = signals['open_time'].values

        keep_cols = ['event_id', 'event_type', 'signal_offset', 'signal_id']
        for c in keep_cols:
            if c in signals.columns:
                features_df[c] = signals[c].values

        return features_df
