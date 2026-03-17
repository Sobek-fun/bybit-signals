import numpy as np
import pandas as pd
from typing import Optional


class PumpLabelerLookahead:
    def __init__(
            self,
            cooldown_bars: int = 8,
            pullback_lookahead: int = 10,
            squeeze_lookahead: int = 32,
            base_pullback_pct: float = 0.05,
            base_squeeze_pct: float = 0.02,
            k1: float = 2.5,
            k2: float = 1.2
    ):
        self.cooldown_bars = cooldown_bars
        self.pullback_lookahead = pullback_lookahead
        self.squeeze_lookahead = squeeze_lookahead
        self.base_pullback_pct = base_pullback_pct
        self.base_squeeze_pct = base_squeeze_pct
        self.k1 = k1
        self.k2 = k2

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

        candidate_mask = peak_mask & runup_mask

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        atr_norm = atr_14 / close

        pullback_pct = np.maximum(self.base_pullback_pct, self.k1 * atr_norm)
        squeeze_pct = np.maximum(self.base_squeeze_pct, self.k2 * atr_norm)

        high_arr = high.values
        low_arr = low.values
        pullback_pct_arr = pullback_pct.values
        squeeze_pct_arr = squeeze_pct.values
        candidate_arr = candidate_mask.values

        candidate_indices = np.nonzero(candidate_arr)[0]

        labels = np.full(n, None, dtype=object)
        runup_values = np.full(n, np.nan, dtype=float)
        last_accepted_index: Optional[int] = None

        for idx in candidate_indices:
            if last_accepted_index is not None and idx - last_accepted_index < self.cooldown_bars:
                continue

            peak_high = high_arr[idx]
            pb_pct = pullback_pct_arr[idx] if not np.isnan(pullback_pct_arr[idx]) else self.base_pullback_pct
            sq_pct = squeeze_pct_arr[idx] if not np.isnan(squeeze_pct_arr[idx]) else self.base_squeeze_pct

            pullback_level = peak_high * (1 - pb_pct)
            squeeze_level = peak_high * (1 + sq_pct)

            pb_end = min(idx + 1 + self.pullback_lookahead, n)
            sq_end = min(idx + 1 + self.squeeze_lookahead, n)

            t_down = None
            for j in range(idx + 1, pb_end):
                if low_arr[j] <= pullback_level:
                    t_down = j
                    break

            t_up = None
            for j in range(idx + 1, sq_end):
                if high_arr[j] >= squeeze_level:
                    t_up = j
                    break

            if t_up is not None and (t_down is None or t_up < t_down):
                label = 'B'
            elif t_down is not None and (t_up is None or t_down < t_up):
                label = 'A'
            else:
                label = 'B'

            labels[idx] = label
            runup_values[idx] = runup.iloc[idx]
            last_accepted_index = idx

        df['pump_la_type'] = labels
        df['pump_la_runup'] = runup_values

        bucket_start = pd.to_datetime(df.index, errors='coerce')

        labeled_mask = df['pump_la_type'].notna()
        df.loc[labeled_mask, 'peak_open_time'] = bucket_start[labeled_mask]
        df.loc[labeled_mask, 'event_open_time'] = bucket_start[labeled_mask] + pd.Timedelta(minutes=15)

        return df
