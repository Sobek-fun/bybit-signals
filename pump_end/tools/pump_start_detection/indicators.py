import pandas as pd
import pandas_ta as ta


class IndicatorCalculator:
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df.ta.mfi(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        return df
