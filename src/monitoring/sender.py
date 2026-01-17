import asyncio
import json
from datetime import datetime
from queue import Queue
from threading import Thread

from aiogram import Bot


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str, ws_broadcaster=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.ws_broadcaster = ws_broadcaster
        self.queue = Queue()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def send_pump_alert(self, symbol: str, close_time: datetime, close_price: float, volume: float):
        message = self._format_message(symbol, close_time, close_price, volume)
        self.queue.put((symbol, close_time, message))

    def send_pump_end_alert(
            self,
            symbol: str,
            signal_open_time: datetime,
            p_end: float,
            threshold: float,
            close_price: float,
            min_pending_bars: int,
            drop_delta: float
    ):
        message = self._format_pump_end_message(
            symbol, signal_open_time, p_end, threshold, close_price, min_pending_bars, drop_delta
        )
        self.queue.put((symbol, signal_open_time, message))

    def _worker(self):
        while True:
            symbol, event_time, message = self.queue.get()
            try:
                asyncio.run(self._send_telegram(message))
                log("INFO", "TG",
                    f"alert sent symbol={symbol} time={event_time.strftime('%Y-%m-%d %H:%M:%S')}")

                if self.ws_broadcaster:
                    payload = {
                        "symbol": symbol,
                        "close_time": event_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "message": message
                    }
                    self.ws_broadcaster.broadcast(json.dumps(payload))
            except Exception as e:
                log("ERROR", "TG",
                    f"send failed symbol={symbol} time={event_time.strftime('%Y-%m-%d %H:%M:%S')} error={type(e).__name__}: {str(e)}")
            finally:
                self.queue.task_done()

    def _format_message(self, symbol: str, close_time: datetime, close_price: float, volume: float) -> str:
        tp_price = close_price * 0.94
        sl_price = close_price * 1.20

        return (
            f"🚀 Strong Pump Detected\n\n"
            f"PUMP DETECTED: {symbol}\n"
            f"Time: {close_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Close: {close_price:.6f}\n"
            f"Volume: {volume:.2f}\n"
            f"TP: {tp_price:.6f} (-6%)\n"
            f"SL: {sl_price:.6f} (+20%)\n\n"
            f"https://www.bybit.com/trade/usdt/{symbol}"
        )

    def _format_pump_end_message(
            self,
            symbol: str,
            signal_open_time: datetime,
            p_end: float,
            threshold: float,
            close_price: float,
            min_pending_bars: int,
            drop_delta: float
    ) -> str:
        tp_price = close_price * 0.94
        sl_price = close_price * 1.20

        return (
            f"📉 Pump End Signal\n\n"
            f"PUMP END: {symbol}\n"
            f"Signal Time: {signal_open_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"p_end: {p_end:.4f} (threshold: {threshold:.4f})\n"
            f"Last Close: {close_price:.6f}\n"
            f"min_pending_bars: {min_pending_bars}\n"
            f"drop_delta: {drop_delta:.4f}\n"
            f"TP: {tp_price:.6f} (-6%)\n"
            f"SL: {sl_price:.6f} (+20%)\n\n"
            f"https://www.bybit.com/trade/usdt/{symbol}"
        )

    async def _send_telegram(self, message: str):
        bot = Bot(token=self.bot_token)
        await bot.send_message(chat_id=self.chat_id, text=message)
        await bot.session.close()
