from typing import Optional

import numpy as np
import pandas as pd

from src.monitoring.pump_params import PumpParams, DEFAULT_PUMP_PARAMS


class PumpDetector:
    def __init__(self, params: PumpParams = None):
        p = params or DEFAULT_PUMP_PARAMS
        self.runup_window = p.runup_window
        self.runup_threshold = p.runup_threshold
        self.context_window = p.context_window
        self.peak_window = p.peak_window
        self.peak_tol = p.peak_tol
        self.volume_median_window = p.volume_median_window
        self.vol_ratio_spike = p.vol_ratio_spike
        self.vol_fade_ratio = p.vol_fade_ratio
        self.corridor_window = p.corridor_window
        self.corridor_quantile = p.corridor_quantile
        self.rsi_hot = p.rsi_hot
        self.mfi_hot = p.mfi_hot
        self.rsi_extreme = p.rsi_extreme
        self.mfi_extreme = p.mfi_extreme
        self.rsi_fade_ratio = p.rsi_fade_ratio
        self.macd_fade_ratio = p.macd_fade_ratio
        self.wick_high = p.wick_high
        self.wick_low = p.wick_low
        self.close_pos_high = p.close_pos_high
        self.close_pos_low = p.close_pos_low
        self.wick_blowoff = p.wick_blowoff
        self.body_blowoff = p.body_blowoff
        self.cooldown_bars = p.cooldown_bars

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df['vol_median'] = df['volume'].rolling(window=self.volume_median_window).median()
        df['vol_ratio'] = df['volume'] / df['vol_median']

        mfi = df['MFI_14']
        rsi = df['RSI_14']
        macdh = df['MACDh_12_26_9']

        macd_line = df['MACD_12_26_9']

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

        close_pos = np.zeros(n, dtype=float)
        np.divide(close_p - low_p, candle_range, out=close_pos, where=range_pos)

        max_oc = np.maximum(open_p, close_p)
        upper_wick = high_p - max_oc

        wick_ratio = np.zeros(n, dtype=float)
        np.divide(upper_wick, candle_range, out=wick_ratio, where=range_pos)

        body_size = np.abs(close_p - open_p)
        body_ratio = np.zeros(n, dtype=float)
        np.divide(body_size, candle_range, out=body_ratio, where=range_pos)

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

        raw_trigger = np.zeros(n, dtype=bool)

        last_star_index: Optional[int] = None
        for i in np.nonzero(strong_cond)[0]:
            if i < skip_initial:
                continue
            if last_star_index is not None and i - last_star_index < self.cooldown_bars:
                continue
            raw_trigger[i] = True
            last_star_index = i

        macd_turn_down = (
                macd_line.notna() &
                macd_line.shift(1).notna() &
                (macd_line < macd_line.shift(1))
        ).to_numpy(dtype=bool, copy=False)

        pending = False
        for t in range(skip_initial, n):
            if raw_trigger[t] and not pending:
                pending = True

            if pending and macd_turn_down[t]:
                signals[t] = 'strong_pump'
                pending = False

        df['pump_score'] = pump_score
        df['pump_signal'] = signals

        df['runup'] = runup
        df['runup_met'] = runup_met
        df['vol_spike_cond'] = vol_spike_cond
        df['vol_spike_recent'] = vol_spike_recent
        df['rsi_hot'] = rsi_hot
        df['mfi_hot'] = mfi_hot
        df['osc_hot_recent'] = osc_hot_recent
        df['macd_pos_recent'] = macd_pos_recent
        df['pump_ctx'] = pump_ctx
        df['near_peak'] = near_peak
        df['blowoff_exhaustion'] = blowoff_exhaustion
        df['predump_peak'] = predump_peak
        df['strong_cond'] = strong_cond

        return df
