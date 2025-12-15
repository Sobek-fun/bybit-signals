from datetime import datetime
from aiogram import Bot
import asyncio


class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    def send_pump_alert(self, symbol: str, timestamp: datetime, close_price: float, volume: float):
        message = self._format_message(symbol, timestamp, close_price, volume)
        asyncio.run(self._send_telegram(message))

    def _format_message(self, symbol: str, timestamp: datetime, close_price: float, volume: float) -> str:
        return (
            f"🚀 Strong Pump Detected\n\n"
            f"Symbol: {symbol}\n"
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Close: {close_price:.6f}\n"
            f"Volume: {volume:.2f}\n\n"
            f"https://www.bybit.com/trade/usdt/{symbol}"
        )

    async def _send_telegram(self, message: str):
        await self.bot.send_message(chat_id=self.chat_id, text=message)
        await self.bot.session.close()
