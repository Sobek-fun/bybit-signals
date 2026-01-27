import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from time import sleep

import pandas as pd

from pump_end_prod.pump_end.feature_builder import PumpFeatureBuilder
from pump_end_prod.infra.clickhouse import DataLoader
from pump_end_prod.infra.logging import log
from pump_end_prod.pump_end.model import PumpEndModel
from pump_end_prod.pump_end.worker import PumpEndWorker, PumpEndWorkerResult, PumpEndSignalState
from pump_end_prod.delivery.telegram_sender import TelegramSender
from pump_end_prod.delivery.ws_broadcaster import WsBroadcaster
from pump_end_prod.delivery.signal_dispatcher import SignalDispatcher


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
            dry_run: bool = False,
            ws_host: str = None,
            ws_port: int = None
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

        self.ws_broadcaster = None
        if ws_host and ws_port:
            self.ws_broadcaster = WsBroadcaster(ws_host, ws_port)
            log("INFO", "PUMP_END", f"WS server started on {ws_host}:{ws_port}")

        self.telegram_sender = TelegramSender(bot_token, chat_id)

        self.signal_dispatcher = SignalDispatcher(
            ws_broadcaster=self.ws_broadcaster,
            telegram_sender=self.telegram_sender,
            dry_run=dry_run
        )

        self.candles_cache: dict[str, pd.DataFrame] = {}
        self.last_processed_minute: datetime = None

        log("INFO", "PUMP_END", f"model loaded: threshold={self.model.threshold:.4f} "
                                f"min_pending_bars={self.model.min_pending_bars} "
                                f"drop_delta={self.model.drop_delta}")

    def run(self):
        log("INFO", "PUMP_END", f"started, waiting for next 15m boundary")

        try:
            while True:
                current_time = datetime.now()

                if self._should_process(current_time):
                    decision_open_time = current_time.replace(second=0, microsecond=0)
                    self._process_cycle(decision_open_time)
                    self.last_processed_minute = decision_open_time

                sleep(0.1)
        except KeyboardInterrupt:
            log("INFO", "PUMP_END", "shutting down...")
        finally:
            if self.ws_broadcaster:
                self.ws_broadcaster.stop()
                log("INFO", "PUMP_END", "WS server stopped")

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
            signal_dispatcher=self.signal_dispatcher
        )
        return worker.process()

    def _log_cycle_summary(self, results: list[PumpEndWorkerResult], cycle_duration: float, load_duration: float):
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        ok = status_counts.get("OK_NO_SIGNAL", 0)
        signals_sent = status_counts.get("SIGNAL_SENT", 0)
        skipped_empty = status_counts.get("SKIP_EMPTY", 0)
        skipped_short = status_counts.get("SKIP_SHORT", 0)
        skipped_missing_last = status_counts.get("SKIP_MISSING_LAST", 0)
        skipped_gapped = status_counts.get("SKIP_GAPPED", 0)
        skipped_features = status_counts.get("SKIP_FEATURES_EMPTY", 0)
        errors = status_counts.get("ERROR", 0)

        log("INFO", "PUMP_END",
            f"cycle done ok={ok} signals_sent={signals_sent} "
            f"skip_empty={skipped_empty} skip_short={skipped_short} "
            f"skip_missing={skipped_missing_last} skip_gapped={skipped_gapped} "
            f"skip_features={skipped_features} errors={errors}")

        if errors > 0:
            error_results = [r for r in results if r.status == "ERROR"]
            first_error = error_results[0]
            log("ERROR", "PUMP_END", f"token={first_error.symbol} error={first_error.error_message}")

        processed_results = [r for r in results if r.status in ("OK_NO_SIGNAL", "SIGNAL_SENT")]

        if processed_results:
            durations = [r.duration_total_ms for r in processed_results]
            features_times = [r.duration_features_ms for r in processed_results]
            predict_times = [r.duration_predict_ms for r in processed_results]

            p50_total = np.percentile(durations, 50)
            p95_total = np.percentile(durations, 95)
            p99_total = np.percentile(durations, 99)
            max_total = max(durations)

            p50_features = np.percentile(features_times, 50)
            p95_features = np.percentile(features_times, 95)

            avg_predict = sum(predict_times) / len(predict_times)

            log("INFO", "PUMP_END",
                f"perf total={cycle_duration:.1f}s load={load_duration:.1f}s "
                f"token_p50={p50_total:.0f}ms p95={p95_total:.0f}ms p99={p99_total:.0f}ms max={max_total:.0f}ms")

            log("INFO", "PUMP_END",
                f"perf features_p50={p50_features:.0f}ms features_p95={p95_features:.0f}ms predict_avg={avg_predict:.1f}ms")

            base_times = [r.duration_base_indicators_ms for r in processed_results]
            pump_times = [r.duration_pump_detector_ms for r in processed_results]
            liq_times = [r.duration_liquidity_ms for r in processed_results]
            shift_times = [r.duration_shift_ms for r in processed_results]
            extract_times = [r.duration_extract_ms for r in processed_results]

            avg_base = sum(base_times) / len(base_times)
            avg_pump = sum(pump_times) / len(pump_times)
            avg_liq = sum(liq_times) / len(liq_times)
            avg_shift = sum(shift_times) / len(shift_times)
            avg_extract = sum(extract_times) / len(extract_times)

            log("INFO", "PUMP_END",
                f"breakdown base={avg_base:.1f}ms pump={avg_pump:.1f}ms liq={avg_liq:.1f}ms "
                f"shift={avg_shift:.1f}ms extract={avg_extract:.1f}ms")

            slowest = sorted(processed_results, key=lambda r: r.duration_total_ms, reverse=True)[:5]
            slowest_str = " ".join([f"{r.symbol}({r.duration_total_ms:.0f}ms)" for r in slowest])
            log("INFO", "PUMP_END", f"slowest: {slowest_str}")

        p_end_values = [r.p_end for r in results if r.p_end is not None]
        if p_end_values:
            above_threshold = sum(1 for p in p_end_values if p >= self.model.threshold)
            max_p_end = max(p_end_values)
            log("INFO", "PUMP_END",
                f"predictions above_threshold={above_threshold}/{len(p_end_values)} max_p_end={max_p_end:.4f}")
