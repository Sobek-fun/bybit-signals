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

        n = len(df)
        if n <= 35:
            return df

        close = df['close']
        low = df['low']
        high = df['high']

        rolling_max_31 = close.rolling(window=31, center=True).max()
        peak_mask = (close == rolling_max_31)

        close_shift5 = close.shift(5)
        runup = (close / close_shift5) - 1
        runup_mask = (runup >= 0.08) & (close_shift5 > 0)

        min_low_next5 = low.shift(-1).rolling(5).min().shift(-4)
        max_high_next5 = high.shift(-1).rolling(5).max().shift(-4)

        candidate_mask = peak_mask & runup_mask

        pullback_threshold = close * 0.97
        squeeze_threshold = close * 1.05

        A_mask = candidate_mask & (min_low_next5 <= pullback_threshold) & (max_high_next5 < squeeze_threshold)
        B_mask = candidate_mask & ~A_mask

        candidate_indices = np.nonzero(candidate_mask.to_numpy())[0]

        labels = np.full(n, None, dtype=object)
        runup_values = np.full(n, np.nan, dtype=float)
        last_accepted_index: Optional[int] = None

        for i in candidate_indices:
            if last_accepted_index is not None and i - last_accepted_index < self.cooldown_bars:
                continue

            if A_mask.iloc[i]:
                labels[i] = 'A'
            else:
                labels[i] = 'B'

            runup_values[i] = runup.iloc[i]
            last_accepted_index = i

        df['pump_la_type'] = labels
        df['pump_la_runup'] = runup_values

        return df
