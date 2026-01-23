from datetime import datetime


def format_pump_alert(symbol: str, close_time: datetime, close_price: float, volume: float) -> str:
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


def format_pump_end_alert(
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
