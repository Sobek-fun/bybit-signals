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
        labels_df['close_time'] = pd.to_datetime(labels_df['timestamp'], utc=True).dt.tz_localize(None)
        labels_df = labels_df.sort_values(['symbol', 'close_time'])

        all_rows = []
        grouped = labels_df.groupby('symbol', sort=False)

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

        self._validate_columns(df)

        return self._extract_features_vectorized(df, symbol, events)

    def _normalize_to_close_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df['close_time'] = df.index + pd.Timedelta(minutes=15)
        if df['close_time'].dt.tz is not None:
            df['close_time'] = df['close_time'].dt.tz_localize(None)
        df = df.set_index('close_time')
        return df

    def _validate_columns(self, df: pd.DataFrame):
        required = ['rsi_14', 'mfi_14', 'macdh_12_26_9']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required indicator columns: {missing}")

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _extract_features_vectorized(self, df: pd.DataFrame, symbol: str, events: pd.DataFrame) -> list:
        event_times = events['close_time'].values
        positions = df.index.get_indexer(pd.DatetimeIndex(event_times))

        required_history = self.warmup_bars + self.window_bars - 1
        valid_mask = (positions >= 0) & (positions >= required_history)

        valid_positions = positions[valid_mask]
        valid_events = events.iloc[valid_mask]

        if len(valid_positions) == 0:
            return []

        base_series = [
            'ret_1', 'range', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'volume', 'log_volume', 'rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio'
        ]
        extended_series = ['atr_14', 'atr_norm', 'bb_z', 'bb_width', 'vwap_dev', 'obv']
        agg_series = ['rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio', 'ret_1']

        series_to_lag = base_series.copy()
        if self.feature_set == "extended":
            series_to_lag.extend([s for s in extended_series if s in df.columns])

        series_arrays = {}
        for s in series_to_lag:
            if s in df.columns:
                series_arrays[s] = df[s].values

        close_arr = df['close'].values
        open_arr = df['open'].values
        volume_arr = df['volume'].values

        num_events = len(valid_positions)
        w = self.window_bars

        lag_matrix = np.zeros((num_events, w), dtype=np.int64)
        for i, pos in enumerate(valid_positions):
            lag_matrix[i] = np.arange(pos, pos - w, -1)

        event_symbols = np.full(num_events, symbol)
        event_close_times = df.index[valid_positions].values
        event_pump_types = valid_events['pump_la_type'].values
        event_targets = (valid_events['pump_la_type'].values == 'A').astype(int)
        event_runups = valid_events['runup_pct'].values

        rows = []
        for ev_idx in range(num_events):
            row = {
                'symbol': event_symbols[ev_idx],
                'close_time': event_close_times[ev_idx],
                'pump_la_type': event_pump_types[ev_idx],
                'target': event_targets[ev_idx],
                'runup_pct': event_runups[ev_idx],
                'timeframe': '15m',
                'window_bars': self.window_bars,
                'warmup_bars': self.warmup_bars
            }

            window_indices = lag_matrix[ev_idx]

            for series_name, arr in series_arrays.items():
                values = arr[window_indices]
                for lag in range(w):
                    row[f'{series_name}_lag_{lag}'] = values[lag]

            for series_name in agg_series:
                if series_name not in series_arrays:
                    continue
                values = series_arrays[series_name][window_indices]

                row[f'{series_name}_max_{w}'] = np.nanmax(values)
                row[f'{series_name}_min_{w}'] = np.nanmin(values)
                row[f'{series_name}_mean_{w}'] = np.nanmean(values)
                row[f'{series_name}_std_{w}'] = np.nanstd(values)
                row[f'{series_name}_last_minus_max_{w}'] = values[0] - np.nanmax(values)

                if len(values) >= 5:
                    row[f'{series_name}_slope_5'] = values[0] - values[4]
                else:
                    row[f'{series_name}_slope_5'] = np.nan

                row[f'{series_name}_delta_1'] = values[0] - values[1] if len(values) >= 2 else np.nan
                row[f'{series_name}_delta_3'] = values[0] - values[2] if len(values) >= 3 else np.nan
                row[f'{series_name}_delta_5'] = values[0] - values[4] if len(values) >= 5 else np.nan

            close_window = close_arr[window_indices]
            row['cum_ret_5'] = (close_window[0] / close_window[4] - 1) if w >= 5 else np.nan
            row['cum_ret_10'] = (close_window[0] / close_window[9] - 1) if w >= 10 else np.nan
            row[f'cum_ret_{w}'] = close_window[0] / close_window[w - 1] - 1

            open_window = open_arr[window_indices]
            red_candles = close_window[:5] < open_window[:5]
            row['count_red_last_5'] = int(np.sum(red_candles))

            upper_wick_last_5 = series_arrays.get('upper_wick_ratio', np.array([]))[window_indices[:5]]
            row['max_upper_wick_last_5'] = np.nanmax(upper_wick_last_5) if len(upper_wick_last_5) > 0 else np.nan

            if 'vol_ratio' in series_arrays:
                vol_ratio_window = series_arrays['vol_ratio'][window_indices]
                row['vol_ratio_max_10'] = np.nanmax(vol_ratio_window[:10]) if w >= 10 else np.nan
                row['vol_ratio_slope_5'] = vol_ratio_window[0] - vol_ratio_window[4] if w >= 5 else np.nan

            volume_window = volume_arr[window_indices]
            max_vol_10 = np.nanmax(volume_window[:10]) if w >= 10 else np.nan
            row['volume_fade'] = volume_window[0] / max_vol_10 if max_vol_10 > 0 else np.nan

            rows.append(row)

        return rows
