from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep

from src.config import Config
from src.monitoring.worker import Worker


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.last_processed_minute = None
        self.last_alerted_bucket = {}

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
        log("INFO", "PIPELINE",
            f"cycle start expected_close_time={expected_close_time.strftime('%Y-%m-%d %H:%M:%S')} tokens={len(self.config.tokens)}")

        cycle_start = datetime.now()
        results = []

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(self._process_token, token): token
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

    def _process_token(self, token: str):
        worker = Worker(self.config, token, self.last_alerted_bucket)
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
        errors = status_counts.get("ERROR", 0)

        log("INFO", "PIPELINE",
            f"cycle done ok={ok} alerts_sent={alerts_sent} dedup={dedup} skipped_empty={skipped_empty} skipped_short={skipped_short} errors={errors}")

        if skipped_short > 0:
            short_examples = [r.symbol for r in results if r.status == "SKIP_SHORT"][:3]
            log("WARN", "PIPELINE",
                f"skipped_short={skipped_short} min_required={Worker.MIN_CANDLES} (examples: {', '.join(short_examples)})")

        if skipped_empty > 0:
            empty_examples = [r.symbol for r in results if r.status == "SKIP_EMPTY"][:3]
            log("WARN", "PIPELINE", f"skipped_empty={skipped_empty} (examples: {', '.join(empty_examples)})")

        durations = [r.duration_total_ms for r in results]
        load_durations = [r.duration_load_ms for r in results if r.duration_load_ms > 0]

        if durations:
            avg_token = sum(durations) / len(durations)
            max_token = max(durations)
            slowest_result = max(results, key=lambda r: r.duration_total_ms)
            slowest = slowest_result.symbol

            load_avg = sum(load_durations) / len(load_durations) if load_durations else 0
            load_max = max(load_durations) if load_durations else 0

            log("INFO", "PIPELINE",
                f"perf total={cycle_duration:.1f}s avg_token={avg_token:.0f}ms max_token={max_token:.0f}ms slowest={slowest} load_avg={load_avg:.0f}ms load_max={load_max:.0f}ms")

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
