from datetime import datetime
from typing import Optional

from pump_end_clustering_prod.delivery.telegram_sender import TelegramSender
from pump_end_clustering_prod.delivery.renderers import format_pump_end_alert


class SignalDispatcher:
    def __init__(
            self,
            telegram_sender: Optional[TelegramSender] = None,
            dry_run: bool = False
    ):
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
            drop_delta: float,
            min_pending_peak: float,
            min_turn_down_bars: int,
            cluster_id: Optional[int] = None
    ):
        if self.telegram_sender and not self.dry_run:
            message = format_pump_end_alert(
                symbol=symbol,
                signal_open_time=event_time,
                p_end=p_end,
                threshold=threshold,
                close_price=close_price,
                min_pending_bars=min_pending_bars,
                drop_delta=drop_delta,
                min_pending_peak=min_pending_peak,
                min_turn_down_bars=min_turn_down_bars,
                cluster_id=cluster_id
            )
            self.telegram_sender.send_message(symbol, event_time, message)
