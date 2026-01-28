import time
import numpy as np
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

from pump_end_prod.infra.clickhouse import DataLoader
from pump_end_prod.pump_end.params import PumpParams, DEFAULT_PUMP_PARAMS


def _process_symbol_worker(args):
    ch_dsn, symbol, events_data, window_bars, warmup_bars, feature_set, params_dict = args

    builder = PumpFeatureBuilder(
        ch_dsn=ch_dsn,
        window_bars=window_bars,
        warmup_bars=warmup_bars,
        feature_set=feature_set,
        params=PumpParams(**params_dict) if params_dict else None
    )

    events = pd.DataFrame(events_data)
    events['open_time'] = pd.to_datetime(events['open_time'])

    return builder._process_symbol(symbol, events)


class PumpFeatureBuilder:
    def __init__(
            self,
            ch_dsn: str = None,
            window_bars: int = 30,
            warmup_bars: int = 150,
            feature_set: str = "base",
            params: PumpParams = None
    ):
        self.ch_dsn = ch_dsn
        self.loader = DataLoader(ch_dsn) if ch_dsn else None
        self.window_bars = window_bars
        self.warmup_bars = warmup_bars
        self.feature_set = feature_set
        self.params = params or DEFAULT_PUMP_PARAMS
        self.vol_ratio_period = 50
        self.vwap_period = 30
        self.corridor_window = self.params.corridor_window
        self.corridor_quantile = self.params.corridor_quantile

    def build(self, labels_df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
        labels_df = labels_df[labels_df['pump_la_type'].isin(['A', 'B'])].copy()

        if 'event_open_time' in labels_df.columns:
            labels_df['open_time'] = pd.to_datetime(labels_df['event_open_time'], utc=True).dt.tz_localize(None)
        elif 'timestamp' in labels_df.columns:
            labels_df['open_time'] = pd.to_datetime(labels_df['timestamp'], utc=True).dt.tz_localize(None)

        labels_df = labels_df.sort_values(['symbol', 'open_time'])

        grouped = labels_df.groupby('symbol', sort=False)
        symbols = list(grouped.groups.keys())

        if len(symbols) <= 2 or max_workers <= 1:
            all_rows = []
            for symbol, group in grouped:
                rows = self._process_symbol(symbol, group)
                all_rows.extend(rows)
        else:
            params_dict = {
                'runup_window': self.params.runup_window,
                'runup_threshold': self.params.runup_threshold,
                'context_window': self.params.context_window,
                'peak_window': self.params.peak_window,
                'peak_tol': self.params.peak_tol,
                'volume_median_window': self.params.volume_median_window,
                'vol_ratio_spike': self.params.vol_ratio_spike,
                'vol_fade_ratio': self.params.vol_fade_ratio,
                'corridor_window': self.params.corridor_window,
                'corridor_quantile': self.params.corridor_quantile,
                'rsi_hot': self.params.rsi_hot,
                'mfi_hot': self.params.mfi_hot,
                'rsi_extreme': self.params.rsi_extreme,
                'mfi_extreme': self.params.mfi_extreme,
                'rsi_fade_ratio': self.params.rsi_fade_ratio,
                'macd_fade_ratio': self.params.macd_fade_ratio,
                'wick_high': self.params.wick_high,
                'wick_low': self.params.wick_low,
                'close_pos_high': self.params.close_pos_high,
                'close_pos_low': self.params.close_pos_low,
                'wick_blowoff': self.params.wick_blowoff,
                'body_blowoff': self.params.body_blowoff,
                'cooldown_bars': self.params.cooldown_bars,
                'liquidity_window_bars': self.params.liquidity_window_bars,
                'eqh_min_touches': self.params.eqh_min_touches,
                'eqh_base_tol': self.params.eqh_base_tol,
                'eqh_atr_factor': self.params.eqh_atr_factor,
            }

            tasks = []
            for symbol, group in grouped:
                events_data = group[['open_time', 'pump_la_type', 'runup_pct']].to_dict('records')
                tasks.append((
                    self.ch_dsn,
                    symbol,
                    events_data,
                    self.window_bars,
                    self.warmup_bars,
                    self.feature_set,
                    params_dict
                ))

            all_rows = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for rows in executor.map(_process_symbol_worker, tasks):
                    all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_rows)
        return result_df

    def build_one_for_inference(
            self,
            df_candles: pd.DataFrame,
            symbol: str,
            decision_open_time: pd.Timestamp
    ) -> tuple[dict, dict]:
        timings = {}
        df = df_candles

        expected_bucket_start = decision_open_time - timedelta(minutes=15)

        events = pd.DataFrame([{
            'open_time': expected_bucket_start,
            'pump_la_type': 'A',
            'runup_pct': 0
        }])

        t0 = time.time()
        df = self._calculate_base_indicators(df)
        timings['base_indicators_ms'] = (time.time() - t0) * 1000

        t0 = time.time()
        df = self._calculate_pump_detector_features(df)
        timings['pump_detector_ms'] = (time.time() - t0) * 1000

        t0 = time.time()
        df = self._calculate_liquidity_features(df, events)
        timings['liquidity_ms'] = (time.time() - t0) * 1000

        if self.feature_set == "extended":
            t0 = time.time()
            df = self._calculate_extended_indicators(df)
            timings['extended_ms'] = (time.time() - t0) * 1000

        df.loc[decision_open_time] = np.nan

        t0 = time.time()
        df = self._apply_decision_shift(df)
        timings['shift_ms'] = (time.time() - t0) * 1000

        events_for_extract = pd.DataFrame([{
            'open_time': decision_open_time,
            'pump_la_type': 'A',
            'runup_pct': 0
        }])

        t0 = time.time()
        rows = self._extract_features_vectorized(df, symbol, events_for_extract)
        timings['extract_ms'] = (time.time() - t0) * 1000

        if not rows:
            return {}, timings

        return rows[0], timings

    def build_many_for_inference(
            self,
            df_candles: pd.DataFrame,
            symbol: str,
            decision_open_times: list
    ) -> list:
        if not decision_open_times:
            return []

        df = df_candles.copy()

        events_liq = pd.DataFrame([{
            'open_time': dt - timedelta(minutes=15),
            'pump_la_type': 'A',
            'runup_pct': 0
        } for dt in decision_open_times])

        df = self._calculate_base_indicators(df)
        df = self._calculate_pump_detector_features(df)
        df = self._calculate_liquidity_features(df, events_liq)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        for dt in decision_open_times:
            if dt not in df.index:
                df.loc[dt] = np.nan

        df = df.sort_index()
        df = self._apply_decision_shift(df)

        events_extract = pd.DataFrame([{
            'open_time': dt,
            'pump_la_type': 'A',
            'runup_pct': 0
        } for dt in decision_open_times])

        rows = self._extract_features_vectorized(df, symbol, events_extract)

        return rows

    def _process_symbol(self, symbol: str, events: pd.DataFrame) -> list:
        t_min = events['open_time'].min()
        t_max = events['open_time'].max()

        buffer_bars = self.warmup_bars + self.window_bars + self.params.liquidity_window_bars + 21
        start_bucket = t_min - pd.Timedelta(minutes=buffer_bars * 15)
        end_bucket = t_max

        df = self.loader.load_candles_range(symbol, start_bucket, end_bucket)

        if df.empty:
            return []

        df = self._calculate_base_indicators(df)
        df = self._calculate_pump_detector_features(df)
        df = self._calculate_liquidity_features(df, events)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        df = self._apply_decision_shift(df)

        self._validate_columns(df)

        return self._extract_features_vectorized(df, symbol, events)

    def _validate_columns(self, df: pd.DataFrame):
        required = ['rsi_14', 'mfi_14', 'macdh_12_26_9']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required indicator columns: {missing}")

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ret_1'] = df['close'] / df['close'].shift(1) - 1

        candle_range = df['high'] - df['low']
        df['range'] = candle_range / df['close']

        max_oc = np.maximum(df['open'], df['close'])
        min_oc = np.minimum(df['open'], df['close'])

        df['upper_wick_ratio'] = np.where(
            candle_range > 0,
            (df['high'] - max_oc) / candle_range,
            0
        )
        df['lower_wick_ratio'] = np.where(
            candle_range > 0,
            (min_oc - df['low']) / candle_range,
            0
        )
        df['body_ratio'] = np.where(
            candle_range > 0,
            np.abs(df['close'] - df['open']) / candle_range,
            0
        )

        df['log_volume'] = np.log(df['volume'].replace(0, np.nan))

        vol_median = df['volume'].rolling(window=self.vol_ratio_period).median()
        df['vol_ratio'] = df['volume'] / vol_median

        df.ta.rsi(length=14, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)

        df = df.rename(columns={
            'RSI_14': 'rsi_14',
            'MFI_14': 'mfi_14',
            'MACDh_12_26_9': 'macdh_12_26_9',
            'MACD_12_26_9': 'macd_line',
            'MACDs_12_26_9': 'macd_signal'
        })

        atr_col = [c for c in df.columns if 'ATR' in c and '14' in c]
        if atr_col:
            df = df.rename(columns={atr_col[0]: 'atr_14'})

        rolling_max_close = df['close'].rolling(window=self.window_bars).max()
        df['drawdown'] = (df['close'] - rolling_max_close) / rolling_max_close

        df['rsi_corridor'] = df['rsi_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)
        df['mfi_corridor'] = df['mfi_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)
        df['macdh_corridor'] = df['macdh_12_26_9'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)

        return df

    def _calculate_pump_detector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params

        df['vol_median_pd'] = df['volume'].rolling(window=p.volume_median_window).median()
        df['vol_ratio_pd'] = df['volume'] / df['vol_median_pd']

        df['vol_ratio_max'] = df['vol_ratio_pd'].rolling(window=p.peak_window).max()
        df['rsi_max'] = df['rsi_14'].rolling(window=p.peak_window).max()
        df['macdh_max'] = df['macdh_12_26_9'].rolling(window=p.peak_window).max()
        df['high_max'] = df['high'].rolling(window=p.peak_window).max()

        min_price = df[['low', 'open']].min(axis=1)
        local_min = min_price.rolling(window=p.runup_window).min()
        df['runup'] = (df['high'] / local_min) - 1
        df['runup_met'] = ((local_min > 0) & (df['runup'] >= p.runup_threshold)).astype(int)

        df['vol_spike_cond'] = (df['vol_ratio_pd'] >= p.vol_ratio_spike).astype(int)
        df['vol_spike_recent'] = (df['vol_spike_cond'].rolling(window=p.context_window).sum() > 0).astype(int)

        rsi = df['rsi_14']
        mfi = df['mfi_14']
        macdh = df['macdh_12_26_9']

        rsi_corridor_pd = rsi.rolling(window=p.corridor_window).quantile(p.corridor_quantile)
        mfi_corridor_pd = mfi.rolling(window=p.corridor_window).quantile(p.corridor_quantile)

        df['rsi_hot'] = (
                rsi.notna() & rsi_corridor_pd.notna() & (rsi >= np.maximum(p.rsi_hot, rsi_corridor_pd))).astype(int)
        df['mfi_hot'] = (
                mfi.notna() & mfi_corridor_pd.notna() & (mfi >= np.maximum(p.mfi_hot, mfi_corridor_pd))).astype(int)
        df['osc_hot_recent'] = ((df['rsi_hot'] | df['mfi_hot']).rolling(window=p.context_window).sum() > 0).astype(int)

        df['macd_pos_recent'] = ((macdh.notna() & (macdh > 0)).rolling(window=p.context_window).sum() > 0).astype(int)

        df['pump_ctx'] = (
                df['runup_met'] & df['vol_spike_recent'] & df['osc_hot_recent'] & df['macd_pos_recent']).astype(int)

        df['near_peak'] = (df['high_max'].notna() & (df['high_max'] > 0) & (
                df['high'] >= df['high_max'] * (1 - p.peak_tol))).astype(int)

        candle_range = df['high'] - df['low']
        range_pos = candle_range > 0

        df['close_pos'] = np.where(range_pos, (df['close'] - df['low']) / candle_range, 0)

        max_oc = np.maximum(df['open'], df['close'])
        df['wick_ratio'] = np.where(range_pos, (df['high'] - max_oc) / candle_range, 0)

        body_size = np.abs(df['close'] - df['open'])
        df['body_ratio_pd'] = np.where(range_pos, body_size / candle_range, 0)

        bearish = df['close'] < df['open']

        df['blowoff_exhaustion'] = (
                (df['close_pos'] <= p.close_pos_low) |
                (bearish & (df['close_pos'] <= 0.45)) |
                ((df['wick_ratio'] >= p.wick_blowoff) & (df['body_ratio_pd'] <= p.body_blowoff))
        ).astype(int)

        df['osc_extreme'] = (rsi.notna() & mfi.notna() & (rsi >= p.rsi_extreme) & (mfi >= p.mfi_extreme)).astype(int)
        df['predump_mask'] = (df['osc_extreme'] & (df['close_pos'] >= p.close_pos_high)).astype(int)

        df['vol_fade'] = (df['vol_ratio_max'].notna() & (df['vol_ratio_max'] > 0) & (
                df['vol_ratio_pd'] <= df['vol_ratio_max'] * p.vol_fade_ratio)).astype(int)
        df['rsi_fade'] = (
                df['rsi_max'].notna() & (df['rsi_max'] > 0) & (rsi <= df['rsi_max'] * p.rsi_fade_ratio)).astype(int)
        df['macd_fade'] = (df['macdh_max'].notna() & macdh.notna() & (df['macdh_max'] > 0) & (
                macdh <= df['macdh_max'] * p.macd_fade_ratio)).astype(int)

        wick_high_mask = df['wick_ratio'] >= p.wick_high
        wick_low_mask = (df['wick_ratio'] >= p.wick_low) & (~wick_high_mask)

        df['predump_peak'] = (
                df['predump_mask'].astype(bool) &
                (
                        (wick_high_mask & df['vol_fade'].astype(bool)) |
                        (wick_low_mask & df['vol_fade'].astype(bool) & (
                                df['rsi_fade'].astype(bool) | df['macd_fade'].astype(bool)))
                )
        ).astype(int)

        df['strong_cond'] = (df['pump_ctx'].astype(bool) & df['near_peak'].astype(bool) & (
                df['blowoff_exhaustion'].astype(bool) | df['predump_peak'].astype(bool))).astype(int)

        df['pump_score'] = df['pump_ctx'] + df['near_peak'] + (df['blowoff_exhaustion'] | df['predump_peak']).astype(
            int)

        return df

    def _calculate_liquidity_features(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)

        high_arr = df['high'].values
        close_arr = df['close'].values
        atr_arr = df['atr_14'].values if 'atr_14' in df.columns else np.full(n, np.nan)

        dates = pd.to_datetime(df.index).date
        date_series = pd.Series(dates, index=df.index)
        unique_dates = date_series.unique()

        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        bar_date_idx = np.array([date_to_idx[d] for d in dates])

        daily_highs = np.zeros(len(unique_dates))
        for i, d in enumerate(unique_dates):
            mask = dates == d
            daily_highs[i] = np.nanmax(high_arr[mask])

        prev_daily_highs = np.roll(daily_highs, 1)
        prev_daily_highs[0] = np.nan

        pdh_arr = prev_daily_highs[bar_date_idx]
        df['pdh'] = pdh_arr

        weeks = pd.to_datetime(df.index).isocalendar()
        year_week = list(zip(weeks.year, weeks.week))
        unique_weeks = list(dict.fromkeys(year_week))

        week_to_idx = {w: i for i, w in enumerate(unique_weeks)}
        bar_week_idx = np.array([week_to_idx[w] for w in year_week])

        weekly_highs = np.zeros(len(unique_weeks))
        for i, w in enumerate(unique_weeks):
            mask = np.array([yw == w for yw in year_week])
            weekly_highs[i] = np.nanmax(high_arr[mask])

        prev_weekly_highs = np.roll(weekly_highs, 1)
        prev_weekly_highs[0] = np.nan

        pwh_arr = prev_weekly_highs[bar_week_idx]
        df['pwh'] = pwh_arr

        df['dist_to_pdh'] = np.where(
            ~np.isnan(pdh_arr) & (pdh_arr > 0),
            (pdh_arr - close_arr) / close_arr,
            np.nan
        )
        df['dist_to_pwh'] = np.where(
            ~np.isnan(pwh_arr) & (pwh_arr > 0),
            (pwh_arr - close_arr) / close_arr,
            np.nan
        )

        df['touched_pdh'] = ((high_arr >= pdh_arr * 0.999) & ~np.isnan(pdh_arr)).astype(int)
        df['touched_pwh'] = ((high_arr >= pwh_arr * 0.999) & ~np.isnan(pwh_arr)).astype(int)

        df['sweep_pdh'] = (
                (high_arr > pdh_arr * 1.001) & (close_arr < pdh_arr * 0.999) & ~np.isnan(pdh_arr)
        ).astype(int)
        df['sweep_pwh'] = (
                (high_arr > pwh_arr * 1.001) & (close_arr < pwh_arr * 0.999) & ~np.isnan(pwh_arr)
        ).astype(int)

        sweep_pdh_mask = df['sweep_pdh'].values == 1
        sweep_pwh_mask = df['sweep_pwh'].values == 1

        overshoot_pdh_arr = np.zeros(n)
        overshoot_pwh_arr = np.zeros(n)
        overshoot_pdh_arr[sweep_pdh_mask] = (high_arr[sweep_pdh_mask] - pdh_arr[sweep_pdh_mask]) / pdh_arr[
            sweep_pdh_mask]
        overshoot_pwh_arr[sweep_pwh_mask] = (high_arr[sweep_pwh_mask] - pwh_arr[sweep_pwh_mask]) / pwh_arr[
            sweep_pwh_mask]

        df['overshoot_pdh'] = overshoot_pdh_arr
        df['overshoot_pwh'] = overshoot_pwh_arr

        liq_window = p.liquidity_window_bars
        eqh_tol_base = p.eqh_base_tol
        eqh_atr_factor = p.eqh_atr_factor
        min_touches = p.eqh_min_touches

        eqh_level_arr = np.full(n, np.nan)
        eqh_strength_arr = np.zeros(n, dtype=int)
        eqh_age_arr = np.full(n, np.nan)

        event_times = events['open_time'].values
        event_positions = df.index.get_indexer(pd.DatetimeIndex(event_times))

        needed_indices = set()
        required_history = self.warmup_bars + self.window_bars
        for pos in event_positions:
            if pos >= 0 and pos >= required_history:
                for offset in range(self.window_bars + 1):
                    idx = pos - offset
                    if idx >= liq_window:
                        needed_indices.add(idx)

        for i in sorted(needed_indices):
            if i < liq_window or i >= n:
                continue

            window_highs = high_arr[i - liq_window:i]
            current_atr = atr_arr[i - 1] if i > 0 and not np.isnan(atr_arr[i - 1]) else 0
            current_close = close_arr[i - 1] if i > 0 else close_arr[i]
            tol = max(eqh_tol_base, eqh_atr_factor * current_atr / current_close if current_close > 0 else eqh_tol_base)

            used = np.zeros(liq_window, dtype=bool)
            sorted_indices = np.argsort(window_highs)[::-1]

            best_strength = 0
            best_level = 0.0
            best_age = np.nan

            for idx in sorted_indices:
                if used[idx]:
                    continue
                level = window_highs[idx]
                if level <= 0:
                    continue

                mask = (~used) & (np.abs(window_highs - level) / level <= tol)
                touches_indices = np.where(mask)[0]

                used[mask] = True

                if len(touches_indices) < min_touches:
                    continue

                avg_level = np.mean(window_highs[touches_indices])
                last_touch_age = liq_window - touches_indices.max() - 1
                strength = len(touches_indices)

                if strength > best_strength or (strength == best_strength and avg_level > best_level):
                    best_strength = strength
                    best_level = avg_level
                    best_age = last_touch_age

            if best_strength >= min_touches:
                eqh_level_arr[i] = best_level
                eqh_strength_arr[i] = best_strength
                eqh_age_arr[i] = best_age

        df['eqh_level'] = eqh_level_arr
        df['eqh_strength'] = eqh_strength_arr
        df['eqh_age_bars'] = eqh_age_arr

        eqh_valid = ~np.isnan(eqh_level_arr) & (eqh_level_arr > 0)
        dist_to_eqh_arr = np.full(n, np.nan)
        dist_to_eqh_arr[eqh_valid] = (eqh_level_arr[eqh_valid] - close_arr[eqh_valid]) / close_arr[eqh_valid]
        df['dist_to_eqh'] = dist_to_eqh_arr

        sweep_eqh_arr = (
                (high_arr > eqh_level_arr * 1.001) & (close_arr < eqh_level_arr * 0.999) & eqh_valid
        ).astype(int)
        df['sweep_eqh'] = sweep_eqh_arr

        sweep_eqh_mask = sweep_eqh_arr == 1
        overshoot_eqh_arr = np.zeros(n)
        overshoot_eqh_arr[sweep_eqh_mask] = (high_arr[sweep_eqh_mask] - eqh_level_arr[sweep_eqh_mask]) / eqh_level_arr[
            sweep_eqh_mask]
        df['overshoot_eqh'] = overshoot_eqh_arr

        sweep_pwh = df['sweep_pwh'].values
        sweep_pdh = df['sweep_pdh'].values
        sweep_eqh = sweep_eqh_arr
        dist_pwh = df['dist_to_pwh'].values
        dist_pdh = df['dist_to_pdh'].values
        dist_eqh = dist_to_eqh_arr
        overshoot_pwh = overshoot_pwh_arr
        overshoot_pdh = overshoot_pdh_arr
        overshoot_eqh = overshoot_eqh_arr

        liq_type_pwh = np.zeros(n, dtype=int)
        liq_type_pdh = np.zeros(n, dtype=int)
        liq_type_eqh = np.zeros(n, dtype=int)
        liq_dist = np.full(n, np.nan)
        liq_sweep = np.zeros(n, dtype=int)
        liq_overshoot = np.zeros(n, dtype=float)

        sweep_pwh_bool = sweep_pwh == 1
        sweep_pdh_bool = sweep_pdh == 1
        sweep_eqh_bool = sweep_eqh == 1

        liq_type_pwh[sweep_pwh_bool] = 1
        liq_dist[sweep_pwh_bool] = dist_pwh[sweep_pwh_bool]
        liq_sweep[sweep_pwh_bool] = 1
        liq_overshoot[sweep_pwh_bool] = overshoot_pwh[sweep_pwh_bool]

        pdh_only = sweep_pdh_bool & ~sweep_pwh_bool
        liq_type_pdh[pdh_only] = 1
        liq_dist[pdh_only] = dist_pdh[pdh_only]
        liq_sweep[pdh_only] = 1
        liq_overshoot[pdh_only] = overshoot_pdh[pdh_only]

        eqh_only = sweep_eqh_bool & ~sweep_pwh_bool & ~sweep_pdh_bool
        liq_type_eqh[eqh_only] = 1
        liq_dist[eqh_only] = dist_eqh[eqh_only]
        liq_sweep[eqh_only] = 1
        liq_overshoot[eqh_only] = overshoot_eqh[eqh_only]

        no_sweep = ~sweep_pwh_bool & ~sweep_pdh_bool & ~sweep_eqh_bool

        dist_pwh_safe = np.where((~np.isnan(dist_pwh)) & (dist_pwh > 0), dist_pwh, np.inf)
        dist_pdh_safe = np.where((~np.isnan(dist_pdh)) & (dist_pdh > 0), dist_pdh, np.inf)
        dist_eqh_safe = np.where((~np.isnan(dist_eqh)) & (dist_eqh > 0), dist_eqh, np.inf)

        dist_stack = np.stack([dist_pwh_safe, dist_pdh_safe, dist_eqh_safe], axis=1)
        min_idx = np.argmin(dist_stack, axis=1)
        min_dist = np.min(dist_stack, axis=1)

        has_candidate = no_sweep & (min_dist < np.inf)

        pwh_best = has_candidate & (min_idx == 0)
        pdh_best = has_candidate & (min_idx == 1)
        eqh_best = has_candidate & (min_idx == 2)

        liq_type_pwh[pwh_best] = 1
        liq_type_pdh[pdh_best] = 1
        liq_type_eqh[eqh_best] = 1
        liq_dist[has_candidate] = min_dist[has_candidate]

        df['liq_level_type_pwh'] = liq_type_pwh
        df['liq_level_type_pdh'] = liq_type_pdh
        df['liq_level_type_eqh'] = liq_type_eqh
        df['liq_level_dist'] = liq_dist
        df['liq_sweep_flag'] = liq_sweep
        df['liq_sweep_overshoot'] = liq_overshoot

        close_pos = df['close_pos'].values
        wick_ratio = df['wick_ratio'].values
        df['liq_reject_strength'] = np.where(
            (close_pos <= 0.4) & (wick_ratio >= 0.25),
            (0.4 - close_pos) + (wick_ratio - 0.25),
            0
        )

        return df

    def _calculate_extended_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'atr_14' not in df.columns:
            df.ta.atr(length=14, append=True)
            atr_col = [c for c in df.columns if 'ATR' in c and '14' in c]
            if atr_col:
                df = df.rename(columns={atr_col[0]: 'atr_14'})

        df['atr_norm'] = df['atr_14'] / df['close']

        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_z'] = (df['close'] - sma_20) / std_20
        df['bb_width'] = std_20 / sma_20

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).rolling(window=self.vwap_period).sum()
        cumulative_vol = df['volume'].rolling(window=self.vwap_period).sum()
        vwap = cumulative_tp_vol / cumulative_vol
        df['vwap_dev'] = (df['close'] - vwap) / vwap

        df.ta.obv(append=True)
        df = df.rename(columns={'OBV': 'obv'})

        return df

    def _apply_decision_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        shift_columns = [
            'ret_1', 'range', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'volume', 'log_volume', 'rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio',
            'macd_line', 'macd_signal', 'drawdown',
            'rsi_corridor', 'mfi_corridor', 'macdh_corridor',
            'vol_median_pd', 'vol_ratio_pd', 'vol_ratio_max', 'rsi_max', 'macdh_max', 'high_max',
            'runup', 'runup_met', 'vol_spike_cond', 'vol_spike_recent',
            'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'macd_pos_recent',
            'pump_ctx', 'near_peak', 'close_pos', 'wick_ratio', 'body_ratio_pd',
            'blowoff_exhaustion', 'osc_extreme', 'predump_mask',
            'vol_fade', 'rsi_fade', 'macd_fade', 'predump_peak', 'strong_cond', 'pump_score',
            'pdh', 'pwh', 'dist_to_pdh', 'dist_to_pwh',
            'touched_pdh', 'touched_pwh', 'sweep_pdh', 'sweep_pwh',
            'overshoot_pdh', 'overshoot_pwh',
            'eqh_level', 'eqh_strength', 'eqh_age_bars', 'dist_to_eqh', 'sweep_eqh', 'overshoot_eqh',
            'liq_level_type_pwh', 'liq_level_type_pdh', 'liq_level_type_eqh',
            'liq_level_dist', 'liq_sweep_flag', 'liq_sweep_overshoot', 'liq_reject_strength'
        ]

        extended_columns = ['atr_14', 'atr_norm', 'bb_z', 'bb_width', 'vwap_dev', 'obv']

        for col in shift_columns:
            if col in df.columns:
                df[col] = df[col].shift(1)

        if self.feature_set == "extended":
            for col in extended_columns:
                if col in df.columns:
                    df[col] = df[col].shift(1)

        return df

    def _extract_features_vectorized(self, df: pd.DataFrame, symbol: str, events: pd.DataFrame) -> list:
        event_times = events['open_time'].values
        positions = df.index.get_indexer(pd.DatetimeIndex(event_times))

        required_history = self.warmup_bars + self.window_bars
        valid_mask = (positions >= 0) & (positions >= required_history)

        valid_positions = positions[valid_mask]
        valid_events = events.iloc[valid_mask]

        if len(valid_positions) == 0:
            return []

        lag_series = [
            'ret_1', 'vol_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio', 'range',
            'close_pos', 'wick_ratio',
            'liq_sweep_flag', 'liq_sweep_overshoot', 'liq_reject_strength'
        ]

        compact_series = ['rsi_14', 'mfi_14', 'macdh_12_26_9', 'macd_line', 'vol_ratio', 'ret_1', 'drawdown']

        pump_detector_features = [
            'runup', 'runup_met', 'vol_spike_cond', 'vol_spike_recent',
            'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'macd_pos_recent',
            'pump_ctx', 'near_peak', 'blowoff_exhaustion',
            'osc_extreme', 'predump_mask', 'vol_fade', 'rsi_fade', 'macd_fade',
            'predump_peak', 'strong_cond', 'pump_score'
        ]

        liquidity_features = [
            'dist_to_pdh', 'dist_to_pwh', 'dist_to_eqh',
            'touched_pdh', 'touched_pwh', 'sweep_pdh', 'sweep_pwh', 'sweep_eqh',
            'overshoot_pdh', 'overshoot_pwh', 'overshoot_eqh',
            'eqh_strength', 'eqh_age_bars',
            'liq_level_type_pwh', 'liq_level_type_pdh', 'liq_level_type_eqh',
            'liq_level_dist'
        ]

        extended_features = ['atr_norm', 'bb_z', 'bb_width', 'vwap_dev']

        series_arrays = {}
        for s in lag_series:
            if s in df.columns:
                series_arrays[s] = df[s].values

        for s in compact_series:
            if s in df.columns:
                series_arrays[s] = df[s].values

        for s in pump_detector_features:
            if s in df.columns:
                series_arrays[s] = df[s].values

        for s in liquidity_features:
            if s in df.columns:
                series_arrays[s] = df[s].values

        if self.feature_set == "extended":
            for s in extended_features:
                if s in df.columns:
                    series_arrays[s] = df[s].values

        close_arr = df['close'].values
        open_arr = df['open'].values
        volume_arr = df['volume'].values

        corridor_arrays = {}
        for name in ['rsi_corridor', 'mfi_corridor', 'macdh_corridor']:
            if name in df.columns:
                corridor_arrays[name] = df[name].values

        num_events = len(valid_positions)
        w = self.window_bars

        lag_matrix = np.zeros((num_events, w), dtype=np.int64)
        for i, pos in enumerate(valid_positions):
            lag_matrix[i] = np.arange(pos, pos - w, -1)

        event_symbols = np.full(num_events, symbol)
        event_open_times = df.index[valid_positions].values
        event_pump_types = valid_events['pump_la_type'].values
        event_targets = (valid_events['pump_la_type'].values == 'A').astype(int)
        event_runups = valid_events['runup_pct'].values

        rows = []
        for ev_idx in range(num_events):
            row = {
                'symbol': event_symbols[ev_idx],
                'open_time': event_open_times[ev_idx],
                'pump_la_type': event_pump_types[ev_idx],
                'target': event_targets[ev_idx],
                'runup_pct': event_runups[ev_idx],
                'timeframe': '15m',
                'window_bars': self.window_bars,
                'warmup_bars': self.warmup_bars
            }

            window_indices = lag_matrix[ev_idx]

            for series_name in lag_series:
                if series_name not in series_arrays:
                    continue
                values = series_arrays[series_name][window_indices]
                for lag in range(w):
                    row[f'{series_name}_lag_{lag}'] = values[lag]

            for series_name in compact_series:
                if series_name not in series_arrays:
                    continue
                values = series_arrays[series_name][window_indices]

                row[f'{series_name}_max_{w}'] = np.nanmax(values)
                row[f'{series_name}_min_{w}'] = np.nanmin(values)
                row[f'{series_name}_mean_{w}'] = np.nanmean(values)
                row[f'{series_name}_std_{w}'] = np.nanstd(values)
                row[f'{series_name}_last_minus_max_{w}'] = values[0] - np.nanmax(values)

                if len(values) >= 5:
                    row[f'{series_name}_slope_5'] = values[0] - values[4]
                else:
                    row[f'{series_name}_slope_5'] = np.nan

                row[f'{series_name}_delta_1'] = values[0] - values[1] if len(values) >= 2 else np.nan
                row[f'{series_name}_delta_3'] = values[0] - values[2] if len(values) >= 3 else np.nan
                row[f'{series_name}_delta_5'] = values[0] - values[4] if len(values) >= 5 else np.nan

            for corridor_name, base_name in [('rsi_corridor', 'rsi_14'), ('mfi_corridor', 'mfi_14'),
                                             ('macdh_corridor', 'macdh_12_26_9')]:
                if corridor_name in corridor_arrays and base_name in series_arrays:
                    corridor_val = corridor_arrays[corridor_name][window_indices[0]]
                    base_val = series_arrays[base_name][window_indices[0]]
                    row[f'{base_name}_minus_corridor'] = base_val - corridor_val if pd.notna(base_val) and pd.notna(
                        corridor_val) else np.nan

            for feat_name in pump_detector_features:
                if feat_name in series_arrays:
                    row[feat_name] = series_arrays[feat_name][window_indices[0]]

            for feat_name in liquidity_features:
                if feat_name in series_arrays:
                    row[feat_name] = series_arrays[feat_name][window_indices[0]]

            if self.feature_set == "extended":
                for feat_name in extended_features:
                    if feat_name in series_arrays:
                        row[feat_name] = series_arrays[feat_name][window_indices[0]]

            pos = valid_positions[ev_idx]
            if pos >= 6:
                close_prev = close_arr[pos - 1]
                close_5_ago = close_arr[pos - 5]
                row['cum_ret_5'] = (close_prev / close_5_ago - 1) if close_5_ago != 0 else np.nan
            else:
                row['cum_ret_5'] = np.nan

            if pos >= 11:
                close_prev = close_arr[pos - 1]
                close_10_ago = close_arr[pos - 10]
                row['cum_ret_10'] = (close_prev / close_10_ago - 1) if close_10_ago != 0 else np.nan
            else:
                row['cum_ret_10'] = np.nan

            if pos >= w + 1:
                close_prev = close_arr[pos - 1]
                close_w_ago = close_arr[pos - w]
                row[f'cum_ret_{w}'] = (close_prev / close_w_ago - 1) if close_w_ago != 0 else np.nan
            else:
                row[f'cum_ret_{w}'] = np.nan

            if pos >= 6:
                close_slice = close_arr[pos - 5:pos]
                open_slice = open_arr[pos - 5:pos]
                red_candles = close_slice < open_slice
                row['count_red_last_5'] = int(np.sum(red_candles))
            else:
                row['count_red_last_5'] = 0

            if 'upper_wick_ratio' in series_arrays and w >= 5:
                upper_wick_last_5 = series_arrays['upper_wick_ratio'][window_indices[:5]]
                row['max_upper_wick_last_5'] = np.nanmax(upper_wick_last_5) if len(upper_wick_last_5) > 0 else np.nan
            else:
                row['max_upper_wick_last_5'] = np.nan

            if 'vol_ratio' in series_arrays:
                vol_ratio_window = series_arrays['vol_ratio'][window_indices]
                row['vol_ratio_max_10'] = np.nanmax(vol_ratio_window[:10]) if w >= 10 else np.nan
                row['vol_ratio_slope_5'] = vol_ratio_window[0] - vol_ratio_window[4] if w >= 5 else np.nan

            if pos >= 11:
                volume_slice = volume_arr[pos - 10:pos]
                max_vol_10 = np.nanmax(volume_slice)
                vol_prev = volume_arr[pos - 1]
                row['volume_fade'] = vol_prev / max_vol_10 if max_vol_10 > 0 else np.nan
            else:
                row['volume_fade'] = np.nan

            rows.append(row)

        return rows
