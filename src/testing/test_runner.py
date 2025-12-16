import csv
from datetime import datetime, timedelta

from src.config import Config
from src.monitoring.data_loader import DataLoader
from src.monitoring.indicator_calculator import IndicatorCalculator
from src.monitoring.pump_detector import PumpDetector


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class TestRunner:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config.ch_dsn, config.offset_seconds)
        self.calculator = IndicatorCalculator()
        self.detector = PumpDetector()
        self.csv_file_handle = None
        self.csv_writer = None

    def _init_csv_file(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals_test_{timestamp}.csv"
        self.csv_file_handle = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(['timestamp', 'symbol'])
        self.csv_file_handle.flush()
        return filename

    def _write_signal_to_csv(self, close_time: datetime, symbol: str):
        timestamp_str = close_time.strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([timestamp_str, symbol])
        self.csv_file_handle.flush()

    def _close_csv_file(self):
        if self.csv_file_handle:
            self.csv_file_handle.close()

    def run_test(self):
        csv_filename = self._init_csv_file()

        end_close_time = self._get_last_closed_time()
        if end_close_time is None:
            log("ERROR", "TEST", "no closed candles found")
            self._close_csv_file()
            return

        start_close_time = end_close_time - timedelta(days=self.config.test_days)

        end_bucket = end_close_time - timedelta(minutes=15)
        start_bucket = start_close_time - timedelta(minutes=15)
        query_start_bucket = start_bucket - timedelta(minutes=self.config.lookback_candles * 15)

        log("INFO", "TEST",
            f"period close_time: start={start_close_time.strftime('%Y-%m-%d %H:%M:%S')} end={end_close_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log("INFO", "TEST",
            f"query start_bucket={query_start_bucket.strftime('%Y-%m-%d %H:%M:%S')} end_bucket={end_bucket.strftime('%Y-%m-%d %H:%M:%S')}")

        total_signals = 0
        total_errors = 0
        total_symbols = 0

        for token in self.config.tokens:
            symbol = f"{token}USDT"
            signals, errors = self._test_symbol(symbol, query_start_bucket, end_bucket, start_bucket)
            total_signals += signals
            total_errors += errors
            total_symbols += 1

        log("INFO", "TEST", f"done total_symbols={total_symbols} total_signals={total_signals} errors={total_errors}")
        log("INFO", "TEST", f"signals saved to {csv_filename}")

        self._close_csv_file()

    def _test_symbol(self, symbol: str, query_start_bucket: datetime, end_bucket: datetime, start_bucket: datetime):
        df_all = self.loader.load_candles_range(symbol, query_start_bucket, end_bucket)

        if df_all.empty:
            log("WARN", "TEST", f"symbol={symbol} skip: candles_loaded=0")
            return 0, 0

        log("INFO", "TEST", f"symbol={symbol} candles_loaded={len(df_all)}")

        if len(df_all) < self.config.lookback_candles:
            log("WARN", "TEST",
                f"symbol={symbol} skip: candles_loaded={len(df_all)} < lookback={self.config.lookback_candles}")
            return 0, 0

        lookback = self.config.lookback_candles
        signals_found = 0
        lookahead_errors = 0
        windows_total = 0

        start_index = self._find_start_index(df_all, start_bucket, lookback)

        for i in range(start_index, len(df_all)):
            window = df_all.iloc[i - lookback + 1: i + 1].copy()

            window = self.calculator.calculate(window)
            window = self.detector.detect(window)

            last = window.iloc[-1]
            bucket_time = window.index[-1]
            close_time = bucket_time + timedelta(minutes=15)

            if last['pump_signal'] == 'strong_pump':
                log("INFO", "TEST",
                    f"SIGNAL symbol={symbol} close_time={close_time.strftime('%Y-%m-%d %H:%M:%S')} close={last['close']:.6f} volume={last['volume']:.2f}")
                self._write_signal_to_csv(close_time, symbol)
                signals_found += 1

                if self._check_lookahead(df_all, i, lookback, bucket_time, last['pump_signal'], symbol, close_time):
                    lookahead_errors += 1

            if windows_total % 500 == 0 and windows_total > 0:
                if self._check_lookahead(df_all, i, lookback, bucket_time, last.get('pump_signal'), symbol, close_time):
                    lookahead_errors += 1

            windows_total += 1

            if windows_total % 500 == 0:
                log("INFO", "TEST",
                    f"symbol={symbol} progress i={i}/{len(df_all)} windows={windows_total} found={signals_found}")

        log("INFO", "TEST", f"symbol={symbol} done windows={windows_total} signals={signals_found}")

        return signals_found, lookahead_errors

    def _find_start_index(self, df_all, start_bucket, lookback):
        for i in range(lookback - 1, len(df_all)):
            if df_all.index[i] >= start_bucket:
                return i
        return lookback - 1

    def _check_lookahead(self, df_all, i, lookback, bucket_time, original_signal, symbol, close_time):
        K = 10
        if i + K >= len(df_all):
            return False

        window_future = df_all.iloc[i - lookback + 1: i + 1 + K].copy()
        window_future = self.calculator.calculate(window_future)
        window_future = self.detector.detect(window_future)

        try:
            future_signal = window_future.loc[bucket_time, 'pump_signal']
        except KeyError:
            return False

        if original_signal != future_signal:
            log("ERROR", "TEST",
                f"LOOKAHEAD_MISMATCH symbol={symbol} close_time={close_time.strftime('%Y-%m-%d %H:%M:%S')} original_signal={original_signal} future_signal={future_signal}")
            return True

        return False

    def _get_last_closed_time(self):
        if not self.config.tokens:
            return None

        symbol = f"{self.config.tokens[0]}USDT"
        return self.loader._get_last_closed_time(symbol)
