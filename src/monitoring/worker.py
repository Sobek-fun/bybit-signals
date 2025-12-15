import time
from datetime import timedelta

from src.config import Config, WorkerResult
from src.monitoring.data_loader import DataLoader
from src.monitoring.indicator_calculator import IndicatorCalculator
from src.monitoring.pump_detector import PumpDetector
from src.monitoring.telegram_sender import TelegramSender


class Worker:
    MIN_CANDLES = 60

    def __init__(self, config: Config, token: str, last_alerted_bucket: dict):
        self.config = config
        self.token = token
        self.symbol = f"{token}USDT"
        self.last_alerted_bucket = last_alerted_bucket

    def process(self) -> WorkerResult:
        start_time = time.time()

        try:
            load_start = time.time()
            loader = DataLoader(self.config.ch_dsn, self.config.offset_seconds)
            df = loader.load_candles(self.symbol, self.config.lookback_candles)
            load_duration = (time.time() - load_start) * 1000

            if df.empty:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_EMPTY",
                    duration_total_ms=total_duration,
                    duration_load_ms=load_duration
                )

            if len(df) < self.MIN_CANDLES:
                total_duration = (time.time() - start_time) * 1000
                return WorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_SHORT",
                    duration_total_ms=total_duration,
                    duration_load_ms=load_duration,
                    candles_count=len(df)
                )

            indicators_start = time.time()
            calculator = IndicatorCalculator()
            df = calculator.calculate(df)
            indicators_duration = (time.time() - indicators_start) * 1000

            detect_start = time.time()
            detector = PumpDetector()
            df = detector.detect(df)
            detect_duration = (time.time() - detect_start) * 1000

            last_candle = df.iloc[-1]
            bucket = last_candle.name

            if last_candle['pump_signal'] == 'strong_pump':
                if self.last_alerted_bucket.get(self.symbol) == bucket:
                    total_duration = (time.time() - start_time) * 1000
                    return WorkerResult(
                        token=self.token,
                        symbol=self.symbol,
                        status="ALERT_DEDUP",
                        duration_total_ms=total_duration,
                        duration_load_ms=load_duration,
                        duration_indicators_ms=indicators_duration,
                        duration_detect_ms=detect_duration,
                        candles_count=len(df),
                        last_bucket=bucket
                    )

                telegram_start = time.time()
                close_time = bucket + timedelta(minutes=15)
                sender = TelegramSender(self.config.bot_token, self.config.chat_id)
                sender.send_pump_alert(
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
                    duration_load_ms=load_duration,
                    duration_indicators_ms=indicators_duration,
                    duration_detect_ms=detect_duration,
                    duration_telegram_ms=telegram_duration,
                    candles_count=len(df),
                    last_bucket=bucket
                )

            total_duration = (time.time() - start_time) * 1000
            return WorkerResult(
                token=self.token,
                symbol=self.symbol,
                status="OK_NO_SIGNAL",
                duration_total_ms=total_duration,
                duration_load_ms=load_duration,
                duration_indicators_ms=indicators_duration,
                duration_detect_ms=detect_duration,
                candles_count=len(df),
                last_bucket=bucket
            )

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            error_stage = "unknown"
            if "loader" not in locals():
                error_stage = "init"
            elif "df" not in locals() or df is None:
                error_stage = "load"
            elif "calculator" in locals() and "df" in dir() and df is not None:
                if "MFI_14" not in df.columns:
                    error_stage = "indicators"
                else:
                    error_stage = "detect"
            elif "sender" in locals():
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
