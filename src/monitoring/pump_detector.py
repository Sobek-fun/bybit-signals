from typing import Optional

import numpy as np
import pandas as pd


class PumpDetector:
    def __init__(
            self,
            runup_window: int = 8,
            runup_threshold: float = 0.08,
            context_window: int = 16,
            peak_window: int = 8,
            peak_tol: float = 0.005,
            volume_median_window: int = 20,
            vol_ratio_spike: float = 5.0,
            vol_fade_ratio: float = 0.85,
            corridor_window: int = 30,
            corridor_quantile: float = 0.90,
            rsi_hot: float = 75.0,
            mfi_hot: float = 80.0,
            rsi_extreme: float = 85.0,
            mfi_extreme: float = 85.0,
            rsi_fade_ratio: float = 0.98,
            macd_fade_ratio: float = 0.99,
            wick_high: float = 0.28,
            wick_low: float = 0.20,
            close_pos_high: float = 0.60,
            close_pos_low: float = 0.35,
            wick_blowoff: float = 0.35,
            body_blowoff: float = 0.25,
            cooldown_bars: int = 8
    ):
        self.runup_window = runup_window
        self.runup_threshold = runup_threshold
        self.context_window = context_window
        self.peak_window = peak_window
        self.peak_tol = peak_tol
        self.volume_median_window = volume_median_window
        self.vol_ratio_spike = vol_ratio_spike
        self.vol_fade_ratio = vol_fade_ratio
        self.corridor_window = corridor_window
        self.corridor_quantile = corridor_quantile
        self.rsi_hot = rsi_hot
        self.mfi_hot = mfi_hot
        self.rsi_extreme = rsi_extreme
        self.mfi_extreme = mfi_extreme
        self.rsi_fade_ratio = rsi_fade_ratio
        self.macd_fade_ratio = macd_fade_ratio
        self.wick_high = wick_high
        self.wick_low = wick_low
        self.close_pos_high = close_pos_high
        self.close_pos_low = close_pos_low
        self.wick_blowoff = wick_blowoff
        self.body_blowoff = body_blowoff
        self.cooldown_bars = cooldown_bars

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df['vol_median'] = df['volume'].rolling(window=self.volume_median_window).median()
        df['vol_ratio'] = df['volume'] / df['vol_median']

        mfi = df['MFI_14']
        rsi = df['RSI_14']
        macdh = df['MACDh_12_26_9']

        df['MFI_corridor'] = mfi.rolling(window=self.corridor_window).quantile(self.corridor_quantile).shift(1)
        df['RSI_corridor'] = rsi.rolling(window=self.corridor_window).quantile(self.corridor_quantile).shift(1)
        df['MACDh_corridor'] = macdh.rolling(window=self.corridor_window).quantile(self.corridor_quantile).shift(1)

        df['vol_ratio_max'] = df['vol_ratio'].rolling(window=self.peak_window).max().shift(1)
        df['RSI_max'] = rsi.rolling(window=self.peak_window).max().shift(1)
        df['MACDh_max'] = macdh.rolling(window=self.peak_window).max().shift(1)
        df['high_max'] = df['high'].rolling(window=self.peak_window).max().shift(1)

        df['pump_score'] = 0
        df['pump_signal'] = None

        skip_initial = max(
            self.volume_median_window,
            self.runup_window,
            self.corridor_window,
            self.context_window,
            self.peak_window
        )

        n = len(df)
        if n <= skip_initial:
            return df

        min_price = df[['low', 'open']].min(axis=1)
        local_min = min_price.rolling(window=self.runup_window).min()
        runup = (df['high'] / local_min) - 1
        runup_met = ((local_min > 0) & (runup >= self.runup_threshold)).fillna(False)

        vol_spike_cond = df['vol_ratio'] >= self.vol_ratio_spike
        vol_spike_recent = vol_spike_cond.rolling(window=self.context_window).sum().fillna(0) > 0

        rsi_corridor = df['RSI_corridor']
        mfi_corridor = df['MFI_corridor']
        rsi_hot = rsi.notna() & rsi_corridor.notna() & (rsi >= np.maximum(self.rsi_hot, rsi_corridor))
        mfi_hot = mfi.notna() & mfi_corridor.notna() & (mfi >= np.maximum(self.mfi_hot, mfi_corridor))
        osc_hot_recent = (rsi_hot | mfi_hot).rolling(window=self.context_window).sum().fillna(0) > 0

        macd_pos_recent = (macdh.notna() & (macdh > 0)).rolling(window=self.context_window).sum().fillna(0) > 0

        pump_ctx = runup_met & vol_spike_recent & osc_hot_recent & macd_pos_recent

        high_max = df['high_max']
        near_peak = high_max.notna() & (high_max > 0) & (df['high'] >= high_max * (1 - self.peak_tol))

        open_p = df['open'].to_numpy(dtype=float, copy=False)
        high_p = df['high'].to_numpy(dtype=float, copy=False)
        low_p = df['low'].to_numpy(dtype=float, copy=False)
        close_p = df['close'].to_numpy(dtype=float, copy=False)

        candle_range = high_p - low_p
        range_pos = candle_range > 0

        close_pos = np.where(range_pos, (close_p - low_p) / candle_range, 0.0)
        max_oc = np.where(np.isnan(open_p), open_p, np.where(np.isnan(close_p), open_p, np.maximum(open_p, close_p)))
        upper_wick = high_p - max_oc
        wick_ratio = np.where(range_pos, upper_wick / candle_range, 0.0)
        body_size = np.abs(close_p - open_p)
        body_ratio = np.where(range_pos, body_size / candle_range, 0.0)
        bearish = close_p < open_p

        blowoff_exhaustion = (
                (close_pos <= self.close_pos_low) |
                (bearish & (close_pos <= 0.45)) |
                ((wick_ratio >= self.wick_blowoff) & (body_ratio <= self.body_blowoff))
        )

        osc_extreme = rsi.notna() & mfi.notna() & (rsi >= self.rsi_extreme) & (mfi >= self.mfi_extreme)
        predump_mask = osc_extreme & (close_pos >= self.close_pos_high)

        vol_ratio_max = df['vol_ratio_max']
        vol_ratio = df['vol_ratio']
        vol_fade = vol_ratio_max.notna() & (vol_ratio_max > 0) & (vol_ratio <= vol_ratio_max * self.vol_fade_ratio)

        wick_high_mask = wick_ratio >= self.wick_high
        wick_low_mask = (wick_ratio >= self.wick_low) & (~wick_high_mask)

        rsi_max = df['RSI_max']
        macdh_max = df['MACDh_max']
        rsi_fade = rsi_max.notna() & (rsi_max > 0) & (rsi <= rsi_max * self.rsi_fade_ratio)
        macd_fade = macdh_max.notna() & macdh.notna() & (macdh_max > 0) & (macdh <= macdh_max * self.macd_fade_ratio)

        predump_peak = (
                predump_mask &
                (
                        (wick_high_mask & vol_fade) |
                        (wick_low_mask & vol_fade & (rsi_fade | macd_fade))
                )
        ).fillna(False).to_numpy(dtype=bool, copy=False)

        pump_ctx_arr = pump_ctx.to_numpy(dtype=bool, copy=False)
        near_peak_arr = near_peak.to_numpy(dtype=bool, copy=False)
        strong_cond = pump_ctx_arr & near_peak_arr & (blowoff_exhaustion | predump_peak)

        score_arr = pump_ctx_arr.astype(int) + near_peak_arr.astype(int) + (blowoff_exhaustion | predump_peak).astype(
            int)

        pump_score = np.zeros(n, dtype=int)
        pump_score[skip_initial:] = score_arr[skip_initial:]

        signals = np.full(n, None, dtype=object)

        last_star_index: Optional[int] = None
        for i in np.nonzero(strong_cond)[0]:
            if i < skip_initial:
                continue
            if last_star_index is not None and i - last_star_index < self.cooldown_bars:
                continue
            signals[i] = 'strong_pump'
            last_star_index = i

        df['pump_score'] = pump_score
        df['pump_signal'] = signals

        return df
