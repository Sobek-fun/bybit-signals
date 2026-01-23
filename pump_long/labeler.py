import numpy as np
import pandas as pd
from typing import Optional

from pump_long.pump.labeler_lookahead import PumpLabelerLookahead


class PumpStartLabelerLookahead:
    def __init__(
            self,
            start_search_bars: int = 12,
            min_runup_from_start: float = 0.06,
            lambda_time: float = 0.005,
            lambda_mae: float = 0.5,
            cooldown_bars: int = 8
    ):
        self.start_search_bars = start_search_bars
        self.min_runup_from_start = min_runup_from_start
        self.lambda_time = lambda_time
        self.lambda_mae = lambda_mae
        self.cooldown_bars = cooldown_bars
        self.peak_labeler = PumpLabelerLookahead(cooldown_bars=cooldown_bars)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df_with_peaks = self.peak_labeler.detect(df)

        df['pump_start_type'] = None
        df['pump_start_runup'] = np.nan
        df['start_open_time'] = pd.NaT
        df['peak_open_time'] = pd.NaT

        peak_mask = df_with_peaks['pump_la_type'] == 'A'
        peak_indices = np.nonzero(peak_mask.to_numpy())[0]

        if len(peak_indices) == 0:
            return df

        open_arr = df['open'].values
        high_arr = df['high'].values
        low_arr = df['low'].values

        n = len(df)
        labels = np.full(n, None, dtype=object)
        runup_values = np.full(n, np.nan, dtype=float)
        start_times = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
        peak_times = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')

        bucket_start = pd.to_datetime(df.index, errors='coerce')

        last_accepted_start: Optional[int] = None

        for peak_idx in peak_indices:
            if peak_idx < self.start_search_bars:
                continue

            high_peak = high_arr[peak_idx]

            search_start = peak_idx - self.start_search_bars
            search_end = peak_idx - 1

            best_score = -np.inf
            best_j = None
            best_mfe = None

            for j in range(search_start, search_end + 1):
                open_j = open_arr[j]
                if open_j <= 0:
                    continue

                mfe = (high_peak / open_j) - 1

                if mfe < self.min_runup_from_start:
                    continue

                min_low_between = np.min(low_arr[j:peak_idx + 1])
                mae = (min_low_between / open_j) - 1

                lead = peak_idx - j

                score = mfe - self.lambda_time * lead - self.lambda_mae * abs(mae)

                if score > best_score:
                    best_score = score
                    best_j = j
                    best_mfe = mfe

            if best_j is None:
                continue

            if last_accepted_start is not None and best_j - last_accepted_start < self.cooldown_bars:
                continue

            labels[best_j] = 'A'
            runup_values[best_j] = best_mfe
            start_times[best_j] = bucket_start[best_j]
            peak_times[best_j] = bucket_start[peak_idx]

            last_accepted_start = best_j

        df['pump_start_type'] = labels
        df['pump_start_runup'] = runup_values
        df['start_open_time'] = start_times
        df['peak_open_time'] = peak_times

        return df
