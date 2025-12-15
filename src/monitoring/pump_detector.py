from typing import Optional

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
        df = df.copy()

        df['vol_median'] = df['volume'].rolling(window=self.volume_median_window).median()
        df['vol_ratio'] = df['volume'] / df['vol_median']

        df['MFI_corridor'] = df['MFI_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile).shift(1)
        df['RSI_corridor'] = df['RSI_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile).shift(1)
        df['MACDh_corridor'] = df['MACDh_12_26_9'].rolling(window=self.corridor_window).quantile(
            self.corridor_quantile).shift(1)

        df['vol_ratio_max'] = df['vol_ratio'].rolling(window=self.peak_window).max().shift(1)
        df['RSI_max'] = df['RSI_14'].rolling(window=self.peak_window).max().shift(1)
        df['MACDh_max'] = df['MACDh_12_26_9'].rolling(window=self.peak_window).max().shift(1)
        df['high_max'] = df['high'].rolling(window=self.peak_window).max().shift(1)

        df['pump_score'] = 0
        df['pump_signal'] = None

        last_star_index: Optional[int] = None

        skip_initial = max(
            self.volume_median_window,
            self.runup_window,
            self.corridor_window,
            self.context_window,
            self.peak_window
        )

        for i in range(skip_initial, len(df)):
            runup_met = False
            local_min_price = float('inf')
            for j in range(i - self.runup_window + 1, i + 1):
                if j >= 0:
                    local_min_price = min(local_min_price, df.iloc[j]['low'], df.iloc[j]['open'])
            if local_min_price > 0:
                runup = (df.iloc[i]['high'] / local_min_price) - 1
                if runup >= self.runup_threshold:
                    runup_met = True

            vol_spike_recent = False
            for j in range(max(0, i - self.context_window + 1), i + 1):
                vol_ratio = df.iloc[j]['vol_ratio']
                if not pd.isna(vol_ratio) and vol_ratio >= self.vol_ratio_spike:
                    vol_spike_recent = True
                    break

            osc_hot_recent = False
            for j in range(max(0, i - self.context_window + 1), i + 1):
                rsi = df.iloc[j].get('RSI_14')
                mfi = df.iloc[j].get('MFI_14')
                rsi_corridor = df.iloc[j]['RSI_corridor']
                mfi_corridor = df.iloc[j]['MFI_corridor']

                rsi_hot = not pd.isna(rsi) and not pd.isna(rsi_corridor) and rsi >= max(self.rsi_hot, rsi_corridor)
                mfi_hot = not pd.isna(mfi) and not pd.isna(mfi_corridor) and mfi >= max(self.mfi_hot, mfi_corridor)

                if rsi_hot or mfi_hot:
                    osc_hot_recent = True
                    break

            macd_pos_recent = False
            for j in range(max(0, i - self.context_window + 1), i + 1):
                macdh = df.iloc[j].get('MACDh_12_26_9')
                if not pd.isna(macdh) and macdh > 0:
                    macd_pos_recent = True
                    break

            pump_ctx = runup_met and vol_spike_recent and osc_hot_recent and macd_pos_recent

            near_peak = False
            high_max = df.iloc[i]['high_max']
            if not pd.isna(high_max) and high_max > 0:
                if df.iloc[i]['high'] >= high_max * (1 - self.peak_tol):
                    near_peak = True

            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']

            candle_range = high_price - low_price
            close_pos = (close_price - low_price) / candle_range if candle_range > 0 else 0
            upper_wick = high_price - max(open_price, close_price)
            wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
            body_size = abs(close_price - open_price)
            body_ratio = body_size / candle_range if candle_range > 0 else 0
            bearish = close_price < open_price

            blowoff_exhaustion = False
            if close_pos <= self.close_pos_low:
                blowoff_exhaustion = True
            elif bearish and close_pos <= 0.45:
                blowoff_exhaustion = True
            elif wick_ratio >= self.wick_blowoff and body_ratio <= self.body_blowoff:
                blowoff_exhaustion = True

            predump_peak = False
            rsi = df.iloc[i].get('RSI_14')
            mfi = df.iloc[i].get('MFI_14')
            osc_extreme = not pd.isna(rsi) and rsi >= self.rsi_extreme and not pd.isna(mfi) and mfi >= self.mfi_extreme

            if osc_extreme and close_pos >= self.close_pos_high:
                vol_ratio_max = df.iloc[i]['vol_ratio_max']
                vol_ratio = df.iloc[i]['vol_ratio']
                vol_fade = not pd.isna(
                    vol_ratio_max) and vol_ratio_max > 0 and vol_ratio <= vol_ratio_max * self.vol_fade_ratio

                if wick_ratio >= self.wick_high:
                    if vol_fade:
                        predump_peak = True
                elif wick_ratio >= self.wick_low:
                    rsi_max = df.iloc[i]['RSI_max']
                    macdh_max = df.iloc[i]['MACDh_max']
                    macdh = df.iloc[i].get('MACDh_12_26_9')

                    rsi_fade = not pd.isna(rsi_max) and rsi_max > 0 and rsi <= rsi_max * self.rsi_fade_ratio
                    macd_fade = not pd.isna(macdh_max) and not pd.isna(
                        macdh) and macdh_max > 0 and macdh <= macdh_max * self.macd_fade_ratio

                    if vol_fade and (rsi_fade or macd_fade):
                        predump_peak = True

            score = sum([pump_ctx, near_peak, blowoff_exhaustion or predump_peak])
            df.at[df.index[i], 'pump_score'] = score

            if last_star_index is not None and i - last_star_index < self.cooldown_bars:
                continue

            if pump_ctx and near_peak and (predump_peak or blowoff_exhaustion):
                df.at[df.index[i], 'pump_signal'] = 'strong_pump'
                last_star_index = i

        return df
