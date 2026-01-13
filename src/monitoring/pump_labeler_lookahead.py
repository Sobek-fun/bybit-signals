import pandas as pd
from typing import Optional


class PumpLabelerLookahead:
    def __init__(self, cooldown_bars: int = 8):
        self.cooldown_bars = cooldown_bars

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pump_la_type'] = None

        n = len(df)
        if n <= 35:
            return df

        last_accepted_index: Optional[int] = None

        for i in range(15, n - 15):
            close_i = df.iloc[i]['close']

            window_start = i - 15
            window_end = i + 16
            close_window = df.iloc[window_start:window_end]['close']

            if close_i != close_window.max():
                continue

            close_i_minus_5 = df.iloc[i - 5]['close']
            if close_i_minus_5 <= 0:
                continue

            runup = (close_i / close_i_minus_5) - 1
            if runup < 0.08:
                continue

            if last_accepted_index is not None and i - last_accepted_index < self.cooldown_bars:
                continue

            if i + 5 >= n:
                continue

            next_5_start = i + 1
            next_5_end = i + 6
            next_5_lows = df.iloc[next_5_start:next_5_end]['low']
            next_5_highs = df.iloc[next_5_start:next_5_end]['high']

            min_low = next_5_lows.min()
            max_high = next_5_highs.max()

            pullback_threshold = close_i * 0.97
            squeeze_threshold = close_i * 1.05

            if min_low <= pullback_threshold and max_high < squeeze_threshold:
                df.iloc[i, df.columns.get_loc('pump_la_type')] = 'A'
            else:
                df.iloc[i, df.columns.get_loc('pump_la_type')] = 'B'

            last_accepted_index = i

        return df
