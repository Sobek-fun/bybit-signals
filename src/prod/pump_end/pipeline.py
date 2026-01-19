from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from time import sleep

import pandas as pd

from src.shared.pump_end.feature_builder import PumpFeatureBuilder
from src.shared.clickhouse import DataLoader
from src.shared.logging import log
from src.prod.pump_end.model import PumpEndModel
from src.prod.pump_end.worker import PumpEndWorker, PumpEndWorkerResult, PumpEndSignalState
from src.prod.delivery.telegram_sender import TelegramSender


class PumpEndPipeline:
    MIN_CANDLES = 873

    def __init__(
            self,
            tokens: list[str],
            ch_dsn: str,
            bot_token: str,
            chat_id: str,
            model_dir: str,
            workers: int = 8,
            offset_seconds: int = 3,
            dry_run: bool = False
    ):
        self.tokens = tokens
        self.ch_dsn = ch_dsn
        self.workers = workers
        self.offset_seconds = offset_seconds
        self.dry_run = dry_run

        self.loader = DataLoader(ch_dsn, offset_seconds)
        self.model = PumpEndModel(model_dir)
        self.feature_builder = PumpFeatureBuilder(
            ch_dsn=None,
            window_bars=self.model.window_bars,
            warmup_bars=self.model.warmup_bars,
            feature_set=self.model.feature_set
        )
        self.signal_state = PumpEndSignalState()
        self.telegram_sender = TelegramSender(bot_token, chat_id, ws_broadcaster=None)

        self.candles_cache: dict[str, pd.DataFrame] = {}
        self.last_processed_minute: datetime = None

        log("INFO", "PUMP_END", f"model loaded: threshold={self.model.threshold:.4f} "
                                f"min_pending_bars={self.model.min_pending_bars} "
                                f"drop_delta={self.model.drop_delta}")

    def run(self):
        log("INFO", "PUMP_END", f"started, waiting for next 15m boundary")

        while True:
            current_time = datetime.now()

            if self._should_process(current_time):
                decision_open_time = current_time.replace(second=0, microsecond=0)
                self._process_cycle(decision_open_time)
                self.last_processed_minute = decision_open_time

            sleep(0.1)

    def _should_process(self, current_time: datetime) -> bool:
        if current_time.minute % 15 != 0:
            return False

        if current_time.second < self.offset_seconds:
            return False

        current_minute = current_time.replace(second=0, microsecond=0)
        if self.last_processed_minute == current_minute:
            return False

        return True

    def _process_cycle(self, decision_open_time: datetime):
        expected_bucket_start = decision_open_time - timedelta(minutes=15)
        start_bucket = expected_bucket_start - timedelta(minutes=(self.MIN_CANDLES - 1) * 15)

        log("INFO", "PUMP_END",
            f"cycle start decision_open_time={decision_open_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"last_closed_bucket={expected_bucket_start.strftime('%Y-%m-%d %H:%M:%S')} "
            f"tokens={len(self.tokens)}")

        cycle_start = datetime.now()

        symbols = [f"{token}USDT" for token in self.tokens]

        load_start = datetime.now()

        if not self.candles_cache:
            candles_dict = self.loader.load_candles_batch(symbols, start_bucket, expected_bucket_start)
            self.candles_cache = candles_dict
        else:
            latest_candles = self.loader.load_candles_batch(symbols, expected_bucket_start, expected_bucket_start)
            self._update_cache(latest_candles)
            candles_dict = self.candles_cache

        load_duration = (datetime.now() - load_start).total_seconds()

        results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for token in self.tokens:
                symbol = f"{token}USDT"
                cached_df = candles_dict.get(symbol)
                df_copy = cached_df.copy() if cached_df is not None else None
                futures[executor.submit(
                    self._process_token,
                    token,
                    df_copy,
                    expected_bucket_start
                )] = token

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    token = futures[future]
                    log("ERROR", "PUMP_END", f"token={token}USDT unexpected error: {e}")

        cycle_duration = (datetime.now() - cycle_start).total_seconds()

        self._log_cycle_summary(results, cycle_duration, load_duration)

    def _update_cache(self, latest_candles: dict[str, pd.DataFrame]):
        for symbol, new_df in latest_candles.items():
            if symbol in self.candles_cache and not new_df.empty:
                cached = self.candles_cache[symbol]
                cached = cached.iloc[1:]
                cached = pd.concat([cached, new_df])
                self.candles_cache[symbol] = cached

    def _process_token(
            self,
            token: str,
            df: pd.DataFrame,
            expected_bucket_start: datetime
    ) -> PumpEndWorkerResult:
        if df is None:
            df = pd.DataFrame()

        worker = PumpEndWorker(
            token=token,
            df=df,
            expected_bucket_start=expected_bucket_start,
            model=self.model,
            feature_builder=self.feature_builder,
            signal_state=self.signal_state,
            telegram_sender=self.telegram_sender,
            dry_run=self.dry_run
        )
        return worker.process()

    def _log_cycle_summary(self, results: list[PumpEndWorkerResult], cycle_duration: float, load_duration: float):
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        ok = status_counts.get("OK_NO_SIGNAL", 0)
        signals_sent = status_counts.get("SIGNAL_SENT", 0)
        signals_dry = status_counts.get("SIGNAL_DRY_RUN", 0)
        skipped_empty = status_counts.get("SKIP_EMPTY", 0)
        skipped_short = status_counts.get("SKIP_SHORT", 0)
        skipped_missing_last = status_counts.get("SKIP_MISSING_LAST", 0)
        skipped_gapped = status_counts.get("SKIP_GAPPED", 0)
        skipped_features = status_counts.get("SKIP_FEATURES_EMPTY", 0)
        errors = status_counts.get("ERROR", 0)

        log("INFO", "PUMP_END",
            f"cycle done ok={ok} signals_sent={signals_sent} signals_dry={signals_dry} "
            f"skip_empty={skipped_empty} skip_short={skipped_short} "
            f"skip_missing={skipped_missing_last} skip_gapped={skipped_gapped} "
            f"skip_features={skipped_features} errors={errors}")

        if errors > 0:
            error_results = [r for r in results if r.status == "ERROR"]
            first_error = error_results[0]
            log("ERROR", "PUMP_END", f"token={first_error.symbol} error={first_error.error_message}")

        durations = [r.duration_total_ms for r in results]
        if durations:
            avg_token = sum(durations) / len(durations)
            max_token = max(durations)

            features_times = [r.duration_features_ms for r in results if r.duration_features_ms > 0]
            predict_times = [r.duration_predict_ms for r in results if r.duration_predict_ms > 0]

            avg_features = sum(features_times) / len(features_times) if features_times else 0
            avg_predict = sum(predict_times) / len(predict_times) if predict_times else 0

            log("INFO", "PUMP_END",
                f"perf total={cycle_duration:.1f}s load={load_duration:.1f}s "
                f"avg_features={avg_features:.0f}ms avg_predict={avg_predict:.0f}ms "
                f"avg_token={avg_token:.0f}ms max_token={max_token:.0f}ms")

        p_end_values = [r.p_end for r in results if r.p_end is not None]
        if p_end_values:
            above_threshold = sum(1 for p in p_end_values if p >= self.model.threshold)
            max_p_end = max(p_end_values)
            log("INFO", "PUMP_END",
                f"predictions above_threshold={above_threshold}/{len(p_end_values)} max_p_end={max_p_end:.4f}")
