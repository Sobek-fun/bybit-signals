from datetime import datetime
from typing import Optional


def format_pump_end_alert(
        symbol: str,
        signal_open_time: datetime,
        p_end: float,
        threshold: float,
        close_price: float,
        min_pending_bars: int,
        drop_delta: float,
        min_pending_peak: float,
        min_turn_down_bars: int,
        cluster_id: Optional[int] = None
) -> str:
    tp_price = close_price * 0.94
    sl_price = close_price * 1.20

    cluster_line = ""
    if cluster_id is not None:
        cluster_line = f"Cluster: {cluster_id}\n"

    return (
        f"📉 Pump End Signal (Clustering)\n\n"
        f"PUMP END: {symbol}\n"
        f"Signal Time: {signal_open_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"p_end: {p_end:.4f} (threshold: {threshold:.4f})\n"
        f"Last Close: {close_price:.6f}\n"
        f"{cluster_line}"
        f"min_pending_bars: {min_pending_bars} | drop_delta: {drop_delta:.4f}\n"
        f"min_pending_peak: {min_pending_peak:.4f} | min_turn_down_bars: {min_turn_down_bars}\n"
        f"TP: {tp_price:.6f} (-6%)\n"
        f"SL: {sl_price:.6f} (+20%)\n\n"
        f"https://www.bybit.com/trade/usdt/{symbol}"
    )
