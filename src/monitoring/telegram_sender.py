import asyncio
from datetime import datetime

from aiogram import Bot


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


class TelegramSender:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    def send_pump_alert(self, symbol: str, close_time: datetime, close_price: float, volume: float):
        message = self._format_message(symbol, close_time, close_price, volume)
        try:
            asyncio.run(self._send_telegram(message))
            log("INFO", "TG",
                f"alert sent symbol={symbol} close_time={close_time.strftime('%Y-%m-%d %H:%M:%S')} close={close_price:.6f} volume={volume:.2f}")
        except Exception as e:
            log("ERROR", "TG",
                f"send failed symbol={symbol} close_time={close_time.strftime('%Y-%m-%d %H:%M:%S')} error={type(e).__name__}: {str(e)}")
            raise

    def _format_message(self, symbol: str, close_time: datetime, close_price: float, volume: float) -> str:
        return (
            f"🚀 Strong Pump Detected\n\n"
            f"Symbol: {symbol}\n"
            f"Time: {close_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Close: {close_price:.6f}\n"
            f"Volume: {volume:.2f}\n\n"
            f"https://www.bybit.com/trade/usdt/{symbol}"
        )

    async def _send_telegram(self, message: str):
        await self.bot.send_message(chat_id=self.chat_id, text=message)
        await self.bot.session.close()
