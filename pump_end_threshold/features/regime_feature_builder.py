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
        self._breadth_cache = None
        self._breadth_cache_range = None

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
        warmup = timedelta(hours=8 * 7 * 24)

        btc_candles = self._load_candles_cached('BTCUSDT', t_min - warmup, t_max)
        eth_candles = self._load_candles_cached('ETHUSDT', t_min - warmup, t_max)

        if self.liquid_universe is None:
            self.liquid_universe = get_liquid_universe(
                self.ch_dsn, t_min - timedelta(days=7), t_max, top_n=self.top_n
            )

        breadth_lookback = timedelta(minutes=self.HIGH_8W_BARS * 15 + 96 * 15)
        breadth_candles = self._load_breadth_candles(t_min - breadth_lookback, t_max)

        rows = []
        for i, sig in signals.iterrows():
            row = {}
            ot = sig['open_time']

            row.update(self._detector_confidence_features(sig))

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

            row.update(self._market_interaction_features(btc_features, eth_features, breadth_features, token_features))

            signals_history = signals[signals['open_time'] < ot]
            signals_bucket = signals[signals['open_time'] == ot]
            row.update(self._signal_flow_features(signals_history, sig))
            row['bucket_signals_now'] = len(signals_bucket)
            row['bucket_unique_symbols_now'] = signals_bucket['symbol'].nunique() if len(signals_bucket) > 0 else 0

            if trades_df is not None:
                row.update(self._strategy_state_features(trades_df, ot))

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

    def _detector_confidence_features(self, sig: pd.Series) -> dict:
        f = {}

        core_cols = [
            'p_end_at_fire', 'threshold_gap', 'pending_bars',
            'drop_from_peak_at_fire', 'signal_offset'
        ]
        for col in core_cols:
            f[f'det_{col}'] = sig.get(col, np.nan)

        snapshot_cols = [
            'runup_age', 'pump_ctx_age', 'near_peak_streak',
            'bars_since_new_high', 'close_pos', 'wick_ratio',
            'blowoff_exhaustion', 'predump_peak', 'vol_fade',
            'rsi_fade', 'macd_fade', 'liquidity_distance',
            'sweep_flag', 'reversal_pattern', 'divergence_score',
            'momentum_fade', 'volume_profile_imbalance',
            'order_flow_exhaustion', 'support_level_distance',
            'resistance_break_strength', 'trend_strength_fade',
            'candle_pattern_score', 'market_structure_shift',
            'buying_pressure_fade', 'selling_pressure_spike',
            'breadth_divergence', 'sector_rotation_signal',
            'time_of_day_bias', 'weekly_momentum', 'monthly_momentum',
            'liquidity_grab', 'stop_hunt_signal', 'fakeout_probability',
            'mean_reversion_score', 'breakout_failure_risk',
            'exhaustion_gap', 'island_reversal', 'double_top_proximity',
            'head_shoulders_pattern', 'wedge_pattern_completion'
        ]

        for col in snapshot_cols:
            if col in sig.index:
                f[f'snap_{col}'] = sig[col]

        return f

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
            'token_ret_1', 'token_ret_4', 'token_ret_16', 'token_ret_96',
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
        f[f'{prefix}_within_5pct_high_8w'] = 1 if (high_8w > 0 and c >= high_8w * 0.95) else 0

        high_slice = high[h8w_start:loc + 1]
        if len(high_slice) > 0:
            peak_pos = np.argmax(high_slice)
            f[f'{prefix}_bars_since_high_8w'] = len(high_slice) - 1 - peak_pos
        else:
            f[f'{prefix}_bars_since_high_8w'] = np.nan

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
            breakout_bars = 0
            start = max(0, loc - 15)
            for i in range(start, loc + 1):
                if close[i] >= high_8w * 0.98:
                    breakout_bars += 1
            actual_bars = loc - start + 1
            f[f'{prefix}_breakout_persistence'] = breakout_bars / actual_bars if actual_bars > 0 else 0.0
        else:
            f[f'{prefix}_breakout_persistence'] = 0.0

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

        f[f'{prefix}_vol_expansion'] = 1 if f.get(f'{prefix}_vol_ratio_20', 0) > 3.0 else 0

        return f

    def _fill_asset_nan(self, prefix: str) -> dict:
        keys = [
            f'{prefix}_ret_1', f'{prefix}_ret_4', f'{prefix}_ret_16', f'{prefix}_ret_96',
            f'{prefix}_ret_accel', f'{prefix}_ret_accel_slow',
            f'{prefix}_vol_ratio_20', f'{prefix}_up_streak',
            f'{prefix}_green_frac_16', f'{prefix}_atr_norm',
            f'{prefix}_dist_to_high_8w', f'{prefix}_within_2pct_high_8w',
            f'{prefix}_within_5pct_high_8w', f'{prefix}_bars_since_high_8w',
            f'{prefix}_close_pos', f'{prefix}_upper_wick_ratio', f'{prefix}_lower_wick_ratio',
            f'{prefix}_breakout_persistence', f'{prefix}_dist_to_high_4h',
            f'{prefix}_dist_to_high_24h', f'{prefix}_pullback_last_4h',
            f'{prefix}_vol_expansion',
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
        within_2pct_8w = 0
        within_5pct_8w = 0
        new_high_8w_24h = 0
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

            h8w_start = max(0, loc - self.HIGH_8W_BARS)
            high_8w = np.max(high[h8w_start:loc + 1])
            if high_8w > 0:
                if c >= high_8w * 0.98:
                    within_2pct_8w += 1
                if c >= high_8w * 0.95:
                    within_5pct_8w += 1

                h24_start = max(0, loc - 96)
                if h24_start < loc:
                    high_24h_slice = high[h24_start:loc + 1]
                    if len(high_24h_slice) > 0 and np.max(high_24h_slice) >= high_8w * 0.99:
                        new_high_8w_24h += 1

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
        f['breadth_share_within_2pct_high_8w'] = within_2pct_8w / n
        f['breadth_share_within_5pct_high_8w'] = within_5pct_8w / n
        f['breadth_share_new_high_8w_last_24h'] = new_high_8w_24h / n

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
            'breadth_share_within_2pct_high_8w', 'breadth_share_within_5pct_high_8w',
            'breadth_share_new_high_8w_last_24h',
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
        breadth_top_decile = get_safe(breadth_features, 'breadth_top_decile_ret_16', 0.0)

        if not pd.isna(breadth_mean_16):
            f['token_vs_breadth_mean_16'] = token_ret_16 - breadth_mean_16
            f['token_vs_breadth_median_16'] = token_ret_16 - breadth_median_16

            if not pd.isna(breadth_top_decile):
                f['token_in_top_decile'] = 1 if token_ret_16 >= breadth_top_decile else 0

        token_vol_ratio = get_safe(token_features, 'token_vol_ratio_20', 1.0)
        breadth_vol_spike_share = get_safe(breadth_features, 'breadth_share_vol_spike_20', 0.0)

        if not pd.isna(token_vol_ratio) and not pd.isna(breadth_vol_spike_share):
            f['token_vol_spike_relative'] = 1 if (token_vol_ratio > 5 and breadth_vol_spike_share < 0.2) else 0

        token_near_8w = get_safe(token_features, 'token_within_2pct_high_8w', 0)
        btc_near_8w = get_safe(btc_features, 'btc_within_2pct_high_8w', 0)
        eth_near_8w = get_safe(eth_features, 'eth_within_2pct_high_8w', 0)
        breadth_near_8w = get_safe(breadth_features, 'breadth_share_within_2pct_high_8w', 0)

        if token_near_8w:
            f['token_breakout_vs_market'] = 1 if (
                breadth_near_8w < 0.2 and not btc_near_8w and not eth_near_8w
            ) else 0
        else:
            f['token_breakout_vs_market'] = 0

        return f

    def _market_interaction_features(self, btc_features: dict, eth_features: dict,
                                    breadth_features: dict, token_features: dict) -> dict:
        f = {}

        def get_safe(d, key, default=np.nan):
            return d.get(key, default) if d else default

        btc_ret_16 = get_safe(btc_features, 'btc_ret_16', 0.0)
        eth_ret_16 = get_safe(eth_features, 'eth_ret_16', 0.0)
        token_near_high_96 = get_safe(token_features, 'token_near_high_96', 0)
        token_near_8w = get_safe(token_features, 'token_within_2pct_high_8w', 0)
        token_vol_spike = 1 if get_safe(token_features, 'token_vol_ratio_20', 1) > 5 else 0
        breadth_near_high = get_safe(breadth_features, 'breadth_share_near_high_96', 0)
        breadth_vol_spike = get_safe(breadth_features, 'breadth_share_vol_spike_20', 0)
        btc_near_high = get_safe(btc_features, 'btc_within_2pct_high_8w', 0)
        eth_near_high = get_safe(eth_features, 'eth_within_2pct_high_8w', 0)

        if not pd.isna(btc_ret_16) and not pd.isna(eth_ret_16):
            f['btc_eth_divergence'] = btc_ret_16 - eth_ret_16
            f['btc_strong_eth_weak'] = 1 if (btc_ret_16 > 0.03 and eth_ret_16 < -0.02) else 0
            f['btc_weak_eth_strong'] = 1 if (btc_ret_16 < -0.02 and eth_ret_16 > 0.03) else 0
            f['both_strong'] = 1 if (btc_ret_16 > 0.02 and eth_ret_16 > 0.02) else 0
            f['both_weak'] = 1 if (btc_ret_16 < -0.02 and eth_ret_16 < -0.02) else 0

        if not pd.isna(breadth_near_high):
            f['breadth_near_high_x_btc_near_high'] = breadth_near_high * btc_near_high
            f['breadth_vol_spike_x_token_near_high'] = breadth_vol_spike * token_near_high_96
            f['btc_strong_x_eth_strong_x_token_near_high'] = (
                f.get('both_strong', 0) * token_near_high_96
            )

        token_drawdown = get_safe(token_features, 'token_drawdown', 0.0)
        breadth_mean_16 = get_safe(breadth_features, 'breadth_mean_ret_16', 0.0)
        if not pd.isna(token_drawdown) and not pd.isna(breadth_mean_16):
            f['token_relative_heat'] = (
                1 if (token_near_8w and breadth_mean_16 < 0.02) else 0
            )
            f['token_overheated_vs_breadth'] = (
                1 if (token_near_high_96 and breadth_near_high < 0.2) else 0
            )

        f['extreme_hot_market'] = (
            1 if (breadth_near_high > 0.5 and btc_near_high and eth_near_high) else 0
        )
        f['extreme_vol_spike'] = (
            1 if (breadth_vol_spike > 0.3 and token_vol_spike) else 0
        )

        return f

    def _signal_flow_features(self, signals_before: pd.DataFrame, current_signal: pd.Series) -> dict:
        t = current_signal['open_time']
        curr_symbol = current_signal['symbol']

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
            symbols_4h = signals_4h['symbol'].tolist()
            symbol_counts = signals_4h['symbol'].value_counts()
            f['unique_symbols_last_4h'] = len(symbol_counts)
            f['max_symbol_concentration_4h'] = symbol_counts.max() / len(signals_4h) if len(signals_4h) > 0 else 0.0
        else:
            f['unique_symbols_last_4h'] = 0
            f['max_symbol_concentration_4h'] = 0.0

        if len(signals_24h) > 0:
            symbols_24h = signals_24h['symbol'].tolist()
            f['unique_symbols_last_24h'] = signals_24h['symbol'].nunique()
            f['same_symbol_last_24h'] = (signals_24h['symbol'] == curr_symbol).sum()
        else:
            f['unique_symbols_last_24h'] = 0
            f['same_symbol_last_24h'] = 0

        for w_hours, w_label in [(1, '1h'), (4, '4h'), (24, '24h')]:
            w_time = t - timedelta(hours=w_hours)
            w_signals = signals_before[signals_before['open_time'] > w_time]
            if len(w_signals) > 1:
                times = w_signals['open_time'].sort_values()
                gaps = [(times.iloc[i] - times.iloc[i-1]).total_seconds() / 60 for i in range(1, len(times))]
                f[f'signal_density_median_gap_{w_label}'] = float(np.median(gaps))
                f[f'signal_density_min_gap_{w_label}'] = min(gaps)
            else:
                f[f'signal_density_median_gap_{w_label}'] = np.nan
                f[f'signal_density_min_gap_{w_label}'] = np.nan

        return f

    def _strategy_state_features(self, trades_df: pd.DataFrame, t: datetime) -> dict:
        f = {}

        resolved_before = trades_df[
            (trades_df['exit_time'].notna()) &
            (trades_df['exit_time'] < t) &
            (trades_df['trade_outcome'].isin(['TP', 'SL']))
        ]

        t_24h = t - timedelta(hours=24)
        resolved_24h = resolved_before[resolved_before['exit_time'] > t_24h]

        if len(resolved_24h) > 0:
            n_sl = (resolved_24h['trade_outcome'] == 'SL').sum()
            f['strat_resolved_sl_rate_last_24h'] = n_sl / len(resolved_24h)
            f['strat_resolved_pnl_sum_last_24h'] = float(resolved_24h['pnl_pct'].fillna(0).sum())
        else:
            f['strat_resolved_sl_rate_last_24h'] = np.nan
            f['strat_resolved_pnl_sum_last_24h'] = np.nan

        sl_streak = 0
        if len(resolved_before) > 0:
            for outcome in reversed(resolved_before.sort_values('exit_time')['trade_outcome'].tolist()):
                if outcome == 'SL':
                    sl_streak += 1
                else:
                    break
        f['strat_prev_closed_sl_streak'] = sl_streak

        open_trades = trades_df[
            (trades_df['open_time'] < t) &
            ((trades_df['exit_time'].isna()) | (trades_df['exit_time'] >= t))
        ]
        f['strat_open_trades_now'] = len(open_trades)

        return f

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

        # Keep track of original order
        signals_with_idx = signals.reset_index(drop=True).copy()
        signals_with_idx['_orig_idx'] = range(len(signals_with_idx))
        signals_sorted = signals_with_idx.sort_values('open_time').reset_index(drop=True)

        t_min = signals_sorted['open_time'].min()
        t_max = signals_sorted['open_time'].max()
        warmup = timedelta(hours=8 * 7 * 24)

        btc_candles = self._load_candles_cached('BTCUSDT', t_min - warmup, t_max)
        eth_candles = self._load_candles_cached('ETHUSDT', t_min - warmup, t_max)

        if self.liquid_universe is None:
            self.liquid_universe = get_liquid_universe(
                self.ch_dsn, t_min - timedelta(days=7), t_max, top_n=self.top_n
            )

        breadth_lookback = timedelta(minutes=self.HIGH_8W_BARS * 15 + 96 * 15)
        breadth_candles = self._load_breadth_candles(t_min - breadth_lookback, t_max)

        unique_symbols = signals_sorted['symbol'].unique()
        token_candles_cache = {}
        for sym in unique_symbols:
            token_candles_cache[sym] = self._load_candles_cached(sym, t_min - warmup, t_max)

        all_features = []

        for batch_start in range(0, len(signals_sorted), batch_size):
            batch_end = min(batch_start + batch_size, len(signals_sorted))

            batch_features = []
            for i in range(batch_start, batch_end):
                sig = signals_sorted.iloc[i]

                row = {}
                ot = sig['open_time']

                # Keep track of original index
                row['_orig_idx'] = sig['_orig_idx']

                row.update(self._detector_confidence_features(sig))

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

                row.update(self._market_interaction_features(btc_features, eth_features, breadth_features, token_features))

                signals_history = signals_sorted[signals_sorted['open_time'] < ot]
                signals_bucket = signals_sorted[signals_sorted['open_time'] == ot]
                row.update(self._signal_flow_features(signals_history, sig))
                row['bucket_signals_now'] = len(signals_bucket)
                row['bucket_unique_symbols_now'] = signals_bucket['symbol'].nunique() if len(signals_bucket) > 0 else 0

                if trades_df is not None:
                    row.update(self._strategy_state_features(trades_df, ot))

                btc_loc = btc_candles.index.searchsorted(ot) - 1
                if 0 <= btc_loc < len(btc_candles):
                    row['context_time_used'] = btc_candles.index[btc_loc]
                else:
                    row['context_time_used'] = pd.NaT

                batch_features.append(row)

            all_features.extend(batch_features)

        features_df = pd.DataFrame(all_features)

        # Sort back to original order
        features_df = features_df.sort_values('_orig_idx').reset_index(drop=True)
        features_df = features_df.drop('_orig_idx', axis=1)

        # Essential columns for matching with signals (from original, not sorted)
        if 'symbol' in signals.columns:
            features_df['symbol'] = signals['symbol'].values
        if 'open_time' in signals.columns:
            features_df['open_time'] = signals['open_time'].values

        # Optional columns including signal_offset for proper matching
        keep_cols = ['event_id', 'event_type', 'signal_offset', 'signal_id']
        for c in keep_cols:
            if c in signals.columns:
                features_df[c] = signals[c].values

        return features_df
