from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from time import sleep

import pandas as pd

from src.config import Config
from src.monitoring.data_loader import DataLoader
from src.monitoring.indicator_calculator import IndicatorCalculator
from src.monitoring.pump_detector import PumpDetector
from src.monitoring.telegram_sender import TelegramSender
from src.monitoring.worker import Worker


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.last_processed_minute = None
        self.last_alerted_bucket = {}
        self.loader = DataLoader(self.config.ch_dsn, self.config.offset_seconds)
        self.calculator = IndicatorCalculator()
        self.detector = PumpDetector()
        self.telegram_sender = TelegramSender(self.config.bot_token, self.config.chat_id)

    def run(self):
        while True:
            current_time = datetime.now()

            if self._should_process(current_time):
                self._process_cycle(current_time)
                self.last_processed_minute = current_time.replace(second=0, microsecond=0)

            sleep(0.1)

    def _should_process(self, current_time: datetime) -> bool:
        if current_time.minute % 15 != 0:
            return False

        if current_time.second < self.config.offset_seconds:
            return False

        current_minute = current_time.replace(second=0, microsecond=0)
        if self.last_processed_minute == current_minute:
            return False

        return True

    def _process_cycle(self, current_time: datetime):
        expected_close_time = current_time.replace(second=0, microsecond=0)
        expected_bucket_start = expected_close_time - timedelta(minutes=15)
        start_bucket = expected_bucket_start - timedelta(minutes=(self.config.lookback_candles - 1) * 15)

        log("INFO", "PIPELINE",
            f"cycle start expected_close_time={expected_close_time.strftime('%Y-%m-%d %H:%M:%S')} expected_bucket_start={expected_bucket_start.strftime('%Y-%m-%d %H:%M:%S')} tokens={len(self.config.tokens)}")

        cycle_start = datetime.now()

        symbols = [f"{token}USDT" for token in self.config.tokens]

        start_1m = start_bucket
        end_1m = expected_bucket_start + timedelta(minutes=14)
        candles_1m_dict = self.loader.load_candles_batch(symbols, start_1m, end_1m)

        candles_15m_dict = self._aggregate_1m_to_15m(candles_1m_dict)

        results = []

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(self._process_token, token, candles_15m_dict.get(f"{token}USDT"),
                                expected_bucket_start): token
                for token in self.config.tokens
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    token = futures[future]
                    log("ERROR", "PIPELINE", f"token={token}USDT unexpected error: {e}")

        cycle_duration = (datetime.now() - cycle_start).total_seconds()

        self._log_cycle_summary(results, cycle_duration)

    def _aggregate_1m_to_15m(self, candles_1m_dict: dict) -> dict:
        result = {}
        for symbol, df_1m in candles_1m_dict.items():
            if df_1m.empty:
                result[symbol] = pd.DataFrame()
                continue

            df_1m['bucket15'] = df_1m.index.floor('15min')

            df_15m = df_1m.groupby('bucket15').agg({
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum',
                'buy_volume': 'sum',
                'sell_volume': 'sum',
                'net_volume': 'sum',
                'trades_count': 'sum'
            })

            result[symbol] = df_15m
        return result

    def _process_token(self, token: str, df, expected_bucket_start: datetime):
        if df is None:
            df = pd.DataFrame()

        worker = Worker(self.config, token, df, expected_bucket_start, self.calculator, self.detector,
                        self.last_alerted_bucket, self.telegram_sender)
        return worker.process()

    def _log_cycle_summary(self, results, cycle_duration: float):
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        ok = status_counts.get("OK_NO_SIGNAL", 0)
        alerts_sent = status_counts.get("ALERT_SENT", 0)
        dedup = status_counts.get("ALERT_DEDUP", 0)
        skipped_empty = status_counts.get("SKIP_EMPTY", 0)
        skipped_short = status_counts.get("SKIP_SHORT", 0)
        skipped_missing_last = status_counts.get("SKIP_MISSING_LAST", 0)
        skipped_gapped = status_counts.get("SKIP_GAPPED", 0)
        errors = status_counts.get("ERROR", 0)

        log("INFO", "PIPELINE",
            f"cycle done ok={ok} alerts_sent={alerts_sent} dedup={dedup} skipped_empty={skipped_empty} skipped_short={skipped_short} skipped_missing_last={skipped_missing_last} skipped_gapped={skipped_gapped} errors={errors}")

        if skipped_short > 0:
            short_examples = [r.symbol for r in results if r.status == "SKIP_SHORT"][:3]
            log("WARN", "PIPELINE",
                f"skipped_short={skipped_short} min_required={Worker.MIN_CANDLES} (examples: {', '.join(short_examples)})")

        if skipped_empty > 0:
            empty_examples = [r.symbol for r in results if r.status == "SKIP_EMPTY"][:3]
            log("WARN", "PIPELINE", f"skipped_empty={skipped_empty} (examples: {', '.join(empty_examples)})")

        if skipped_missing_last > 0:
            missing_last_examples = [r.symbol for r in results if r.status == "SKIP_MISSING_LAST"][:3]
            log("WARN", "PIPELINE",
                f"skipped_missing_last={skipped_missing_last} (examples: {', '.join(missing_last_examples)})")

        if skipped_gapped > 0:
            gapped_examples = [r.symbol for r in results if r.status == "SKIP_GAPPED"][:3]
            log("WARN", "PIPELINE",
                f"skipped_gapped={skipped_gapped} (examples: {', '.join(gapped_examples)})")

        durations = [r.duration_total_ms for r in results]

        if durations:
            avg_token = sum(durations) / len(durations)
            max_token = max(durations)
            slowest_result = max(results, key=lambda r: r.duration_total_ms)
            slowest = slowest_result.symbol

            log("INFO", "PIPELINE",
                f"perf total={cycle_duration:.1f}s avg_token={avg_token:.0f}ms max_token={max_token:.0f}ms slowest={slowest}")

        candle_counts = [r.candles_count for r in results if r.candles_count > 0]
        if candle_counts:
            min_candles = min(candle_counts)
            avg_candles = sum(candle_counts) / len(candle_counts)
            max_candles = max(candle_counts)
            log("INFO", "PIPELINE", f"data candles(min/avg/max)={min_candles}/{avg_candles:.0f}/{max_candles}")

        error_results = [r for r in results if r.status == "ERROR"]
        if error_results:
            first_error = error_results[0]
            log("ERROR", "PIPELINE",
                f"token={first_error.symbol} stage={first_error.error_stage} error={first_error.error_type}: {first_error.error_message}")

            if len(error_results) > 1:
                log("ERROR", "PIPELINE", f"same error repeated x{len(error_results)} (showing first only)")
