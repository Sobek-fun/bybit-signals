from datetime import timedelta
from src.config import Config
from src.monitoring.data_loader import DataLoader
from src.monitoring.indicator_calculator import IndicatorCalculator
from src.monitoring.pump_detector import PumpDetector
from src.monitoring.telegram_sender import TelegramSender


class Worker:
    def __init__(self, config: Config, token: str, last_alerted_bucket: dict):
        self.config = config
        self.token = token
        self.symbol = f"{token}USDT"
        self.last_alerted_bucket = last_alerted_bucket

    def process(self):
        loader = DataLoader(self.config.ch_dsn, self.config.offset_seconds)
        df = loader.load_candles(self.symbol, self.config.lookback_candles)

        if df.empty:
            return

        calculator = IndicatorCalculator()
        df = calculator.calculate(df)

        detector = PumpDetector()
        df = detector.detect(df)

        last_candle = df.iloc[-1]

        if last_candle['pump_signal'] == 'strong_pump':
            bucket = last_candle.name

            if self.last_alerted_bucket.get(self.symbol) == bucket:
                return

            close_time = bucket + timedelta(minutes=15)

            sender = TelegramSender(self.config.bot_token, self.config.chat_id)
            sender.send_pump_alert(
                symbol=self.symbol,
                close_time=close_time,
                close_price=last_candle['close'],
                volume=last_candle['volume']
            )

            self.last_alerted_bucket[self.symbol] = bucket
