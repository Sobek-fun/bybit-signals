import json
from datetime import datetime
from typing import Optional

from pump_end_prod.delivery.ws_broadcaster import WsBroadcaster
from pump_end_prod.delivery.telegram_sender import TelegramSender
from pump_end_prod.delivery.renderers import format_pump_end_alert


class SignalDispatcher:
    def __init__(
            self,
            ws_broadcaster: Optional[WsBroadcaster] = None,
            telegram_sender: Optional[TelegramSender] = None,
            dry_run: bool = False
    ):
        self.ws_broadcaster = ws_broadcaster
        self.telegram_sender = telegram_sender
        self.dry_run = dry_run

    def publish_pump_end_signal(
            self,
            symbol: str,
            event_time: datetime,
            p_end: float,
            threshold: float,
            close_price: float,
            min_pending_bars: int,
            drop_delta: float
    ):
        source = "pump_end"
        signal_type = "pump_end_signal"

        signal_data = {
            "source": source,
            "type": signal_type,
            "symbol": symbol,
            "event_time": event_time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": {
                "p_end": p_end,
                "threshold": threshold,
                "close_price": close_price,
                "min_pending_bars": min_pending_bars,
                "drop_delta": drop_delta
            }
        }

        if self.ws_broadcaster:
            self.ws_broadcaster.broadcast(json.dumps(signal_data), source)

        if self.telegram_sender and not self.dry_run:
            message = format_pump_end_alert(
                symbol=symbol,
                signal_open_time=event_time,
                p_end=p_end,
                threshold=threshold,
                close_price=close_price,
                min_pending_bars=min_pending_bars,
                drop_delta=drop_delta
            )
            self.telegram_sender.send_message(symbol, event_time, message)
