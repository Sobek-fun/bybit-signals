import time
from datetime import datetime, timedelta

import pandas as pd

from src.config import Config, WorkerResult
from src.monitoring.indicator_calculator import IndicatorCalculator
from src.monitoring.pump_detector import PumpDetector
from src.monitoring.sender import TelegramSender


class Worker:
    MIN_CANDLES = 60

    def __init__(self, config: Config, token: str, df: pd.DataFrame, expected_bucket_start: datetime,
                 calculator: IndicatorCalculator, detector: PumpDetector, last_alerted_bucket: dict,
                 telegram_sender: TelegramSender):
        self.config = config
        self.token = token
        self.symbol = f"{token}USDT"
        self.df = df
        self.expected_bucket_start = expected_bucket_start
        self.calculator = calculator
        self.detector = detector
        self.last_alerted_bucket = last_alerted_bucket
        self.telegram_sender = telegram_sender

    def process(self) -> WorkerResult:
        start_time = time.time()

        try:
            if self.df.empty:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_EMPTY",
                    duration_total_ms=total_duration
                )

            if self.expected_bucket_start not in self.df.index:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_MISSING_LAST",
                    duration_total_ms=total_duration,
                    candles_count=len(self.df)
                )

            expected_buckets = pd.date_range(
                end=self.expected_bucket_start,
                periods=self.MIN_CANDLES,
                freq='15min'
            )
            missing_buckets = expected_buckets.difference(self.df.index)
            if len(missing_buckets) > 0:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_GAPPED",
                    duration_total_ms=total_duration,
                    candles_count=len(self.df)
                )

            if len(self.df) < self.MIN_CANDLES:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_SHORT",
                    duration_total_ms=total_duration,
                    candles_count=len(self.df)
                )

            indicators_start = time.time()
            self.df = self.calculator.calculate(self.df)
            indicators_duration = (time.time() - indicators_start) * 1000

            detect_start = time.time()
            self.df = self.detector.detect(self.df)
            detect_duration = (time.time() - detect_start) * 1000

            last_candle = self.df.loc[self.expected_bucket_start]
            bucket = self.expected_bucket_start

            if last_candle['pump_signal'] == 'strong_pump':
                if self.last_alerted_bucket.get(self.symbol) == bucket:
                    total_duration = (time.time() - start_time) * 1000
                    return WorkerResult(
                        token=self.token,
                        symbol=self.symbol,
                        status="ALERT_DEDUP",
                        duration_total_ms=total_duration,
                        duration_indicators_ms=indicators_duration,
                        duration_detect_ms=detect_duration,
                        candles_count=len(self.df),
                        last_bucket=bucket
                    )

                telegram_start = time.time()
                close_time = bucket + timedelta(minutes=15)
                self.telegram_sender.send_pump_alert(
                    symbol=self.symbol,
                    close_time=close_time,
                    close_price=last_candle['close'],
                    volume=last_candle['volume']
                )
                telegram_duration = (time.time() - telegram_start) * 1000

                self.last_alerted_bucket[self.symbol] = bucket

                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="ALERT_SENT",
                    duration_total_ms=total_duration,
                    duration_indicators_ms=indicators_duration,
                    duration_detect_ms=detect_duration,
                    duration_telegram_ms=telegram_duration,
                    candles_count=len(self.df),
                    last_bucket=bucket
                )

            total_duration = (time.time() - start_time) * 1000
            return WorkerResult(
                token=self.token,
                symbol=self.symbol,
                status="OK_NO_SIGNAL",
                duration_total_ms=total_duration,
                duration_indicators_ms=indicators_duration,
                duration_detect_ms=detect_duration,
                candles_count=len(self.df),
                last_bucket=bucket
            )

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            error_stage = "unknown"
            if not hasattr(self, 'df') or self.df is None:
                error_stage = "init"
            elif "MFI_14" not in self.df.columns:
                error_stage = "indicators"
            elif 'pump_signal' not in self.df.columns:
                error_stage = "detect"
            else:
                error_stage = "telegram"

            return WorkerResult(
                token=self.token,
                symbol=self.symbol,
                status="ERROR",
                duration_total_ms=total_duration,
                error_stage=error_stage,
                error_type=type(e).__name__,
                error_message=str(e)
            )
