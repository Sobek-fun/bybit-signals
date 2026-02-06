import asyncio
from datetime import datetime
from queue import Queue
from threading import Thread

from aiogram import Bot

from pump_end_clustering_prod.infra.logging import log


class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.queue = Queue()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def send_message(self, symbol: str, event_time: datetime, message: str):
        self.queue.put((symbol, event_time, message))

    def _worker(self):
        while True:
            symbol, event_time, message = self.queue.get()
            try:
                asyncio.run(self._send_telegram(message))
                log("INFO", "TG",
                    f"alert sent symbol={symbol} time={event_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                log("ERROR", "TG",
                    f"send failed symbol={symbol} time={event_time.strftime('%Y-%m-%d %H:%M:%S')} error={type(e).__name__}: {str(e)}")
            finally:
                self.queue.task_done()

    async def _send_telegram(self, message: str):
        bot = Bot(token=self.bot_token)
        await bot.send_message(chat_id=self.chat_id, text=message)
        await bot.session.close()
