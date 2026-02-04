import numpy as np
import pandas as pd
from typing import Optional


class PumpLabelerLookahead:
    def __init__(self, cooldown_bars: int = 8):
        self.cooldown_bars = cooldown_bars

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pump_la_type'] = None
        df['pump_la_runup'] = np.nan
        df['peak_open_time'] = pd.NaT
        df['event_open_time'] = pd.NaT

        n = len(df)
        if n <= 35:
            return df

        close = df['close']
        low = df['low']
        high = df['high']

        rolling_max_high_31 = high.rolling(window=31, center=True).max()
        peak_mask = (high == rolling_max_high_31)

        close_shift5 = close.shift(5)
        runup = (high / close_shift5) - 1
        runup_mask = (runup >= 0.08) & (close_shift5 > 0)

        min_low_next10 = low.shift(-1).rolling(10).min().shift(-9)
        max_high_next10 = high.shift(-1).rolling(10).max().shift(-9)

        candidate_mask = peak_mask & runup_mask

        pullback_threshold = high * 0.97
        squeeze_threshold = high * 1.10

        A_mask = candidate_mask & (min_low_next10 <= pullback_threshold) & (max_high_next10 < squeeze_threshold)
        B_mask = candidate_mask & ~A_mask

        candidate_indices = np.nonzero(candidate_mask.to_numpy())[0]

        labels = np.full(n, None, dtype=object)
        runup_values = np.full(n, np.nan, dtype=float)
        last_accepted_index: Optional[int] = None

        for i in candidate_indices:
            if last_accepted_index is not None and i - last_accepted_index < self.cooldown_bars:
                continue

            labels[i] = 'A' if A_mask.iloc[i] else 'B'
            runup_values[i] = runup.iloc[i]
            last_accepted_index = i

        df['pump_la_type'] = labels
        df['pump_la_runup'] = runup_values

        bucket_start = pd.to_datetime(df.index, errors='coerce')

        labeled_mask = df['pump_la_type'].notna()
        df.loc[labeled_mask, 'peak_open_time'] = bucket_start[labeled_mask]
        df.loc[labeled_mask, 'event_open_time'] = bucket_start[labeled_mask] + pd.Timedelta(minutes=15)

        return df
