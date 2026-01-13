import numpy as np
import pandas as pd
import pandas_ta as ta

from src.monitoring.data_loader import DataLoader


class PumpFeatureBuilder:
    def __init__(
            self,
            ch_dsn: str,
            window_bars: int = 30,
            warmup_bars: int = 150,
            feature_set: str = "base"
    ):
        self.loader = DataLoader(ch_dsn)
        self.window_bars = window_bars
        self.warmup_bars = warmup_bars
        self.feature_set = feature_set
        self.vol_ratio_period = 50
        self.vwap_period = 30

    def build(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        labels_df = labels_df[labels_df['pump_la_type'].isin(['A', 'B'])].copy()
        labels_df['close_time'] = pd.to_datetime(labels_df['timestamp'])
        labels_df = labels_df.sort_values(['symbol', 'close_time'])

        all_rows = []
        grouped = labels_df.groupby('symbol')

        for symbol, group in grouped:
            rows = self._process_symbol(symbol, group)
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_rows)
        return result_df

    def _process_symbol(self, symbol: str, events: pd.DataFrame) -> list:
        t_min = events['close_time'].min()
        t_max = events['close_time'].max()

        buffer_bars = self.warmup_bars + self.window_bars + 20
        start_bucket = t_min - pd.Timedelta(minutes=buffer_bars * 15)
        end_bucket = t_max

        df = self.loader.load_candles_range(symbol, start_bucket, end_bucket)

        if df.empty:
            return []

        df = self._normalize_to_close_time(df)
        df = self._calculate_base_indicators(df)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        rows = []
        for _, event in events.iterrows():
            row = self._extract_features_for_event(df, symbol, event)
            if row is not None:
                rows.append(row)

        return rows

    def _normalize_to_close_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['close_time'] = df.index + pd.Timedelta(minutes=15)
        df = df.set_index('close_time')
        return df

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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

        df = df.rename(columns={
            'RSI_14': 'rsi_14',
            'MFI_14': 'mfi_14',
            'MACDh_12_26_9': 'macdh_12_26_9'
        })

        return df

    def _calculate_extended_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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

    def _extract_features_for_event(self, df: pd.DataFrame, symbol: str, event: pd.Series) -> dict:
        close_time = event['close_time']

        if close_time not in df.index:
            return None

        loc = df.index.get_loc(close_time)
        required_history = self.warmup_bars + self.window_bars

        if loc < required_history - 1:
            return None

        window_start = loc - self.window_bars + 1
        window = df.iloc[window_start:loc + 1].copy()

        if len(window) != self.window_bars:
            return None

        row = {
            'symbol': symbol,
            'close_time': close_time,
            'pump_la_type': event['pump_la_type'],
            'target': 1 if event['pump_la_type'] == 'A' else 0,
            'runup_pct': event['runup_pct'],
            'timeframe': '15m',
            'window_bars': self.window_bars,
            'warmup_bars': self.warmup_bars
        }

        lag_features = self._build_lag_features(window)
        row.update(lag_features)

        agg_features = self._build_aggregate_features(window)
        row.update(agg_features)

        return row

    def _build_lag_features(self, window: pd.DataFrame) -> dict:
        features = {}

        base_series = [
            'ret_1', 'range', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'volume', 'log_volume', 'rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio'
        ]

        extended_series = ['atr_14', 'atr_norm', 'bb_z', 'bb_width', 'vwap_dev', 'obv']

        series_to_lag = base_series.copy()
        if self.feature_set == "extended":
            series_to_lag.extend([s for s in extended_series if s in window.columns])

        for series_name in series_to_lag:
            if series_name not in window.columns:
                continue

            values = window[series_name].values
            for lag in range(self.window_bars):
                idx = self.window_bars - 1 - lag
                features[f'{series_name}_lag_{lag}'] = values[idx]

        return features

    def _build_aggregate_features(self, window: pd.DataFrame) -> dict:
        features = {}

        agg_series = ['rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio', 'ret_1']

        for series_name in agg_series:
            if series_name not in window.columns:
                continue

            values = window[series_name].values

            features[f'{series_name}_max_30'] = np.nanmax(values)
            features[f'{series_name}_min_30'] = np.nanmin(values)
            features[f'{series_name}_mean_30'] = np.nanmean(values)
            features[f'{series_name}_std_30'] = np.nanstd(values)

            features[f'{series_name}_last_minus_max_30'] = values[-1] - np.nanmax(values)

            if len(values) >= 5:
                features[f'{series_name}_slope_5'] = values[-1] - values[-5]
            else:
                features[f'{series_name}_slope_5'] = np.nan

            features[f'{series_name}_delta_1'] = values[-1] - values[-2] if len(values) >= 2 else np.nan
            features[f'{series_name}_delta_3'] = values[-1] - values[-3] if len(values) >= 3 else np.nan
            features[f'{series_name}_delta_5'] = values[-1] - values[-5] if len(values) >= 5 else np.nan

        close_values = window['close'].values
        features['cum_ret_5'] = (close_values[-1] / close_values[-5] - 1) if len(close_values) >= 5 else np.nan
        features['cum_ret_10'] = (close_values[-1] / close_values[-10] - 1) if len(close_values) >= 10 else np.nan
        features['cum_ret_30'] = (close_values[-1] / close_values[0] - 1)

        red_candles = window['close'].values < window['open'].values
        features['count_red_last_5'] = int(np.sum(red_candles[-5:]))

        upper_wick_last_5 = window['upper_wick_ratio'].values[-5:]
        features['max_upper_wick_last_5'] = np.nanmax(upper_wick_last_5)

        if 'vol_ratio' in window.columns:
            vol_ratio_values = window['vol_ratio'].values
            features['vol_ratio_max_10'] = np.nanmax(vol_ratio_values[-10:]) if len(vol_ratio_values) >= 10 else np.nan
            features['vol_ratio_slope_5'] = vol_ratio_values[-1] - vol_ratio_values[-5] if len(
                vol_ratio_values) >= 5 else np.nan

        volume_last_10 = window['volume'].values[-10:]
        max_vol_10 = np.nanmax(volume_last_10)
        features['volume_fade'] = window['volume'].values[-1] / max_vol_10 if max_vol_10 > 0 else np.nan

        return features
