import asyncio
import json
from datetime import datetime
from queue import Queue
from threading import Thread

from aiogram import Bot

from pump_end_prod.infra.logging import log
from pump_end_prod.delivery.renderers import format_pump_alert, format_pump_end_alert


class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str, ws_broadcaster=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.ws_broadcaster = ws_broadcaster
        self.queue = Queue()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def send_pump_alert(self, symbol: str, close_time: datetime, close_price: float, volume: float):
        message = format_pump_alert(symbol, close_time, close_price, volume)
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
        message = format_pump_end_alert(
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

    async def _send_telegram(self, message: str):
        bot = Bot(token=self.bot_token)
        await bot.send_message(chat_id=self.chat_id, text=message)
        await bot.session.close()
