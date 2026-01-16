from dataclasses import dataclass


@dataclass
class PumpParams:
    runup_window: int = 8
    runup_threshold: float = 0.06
    context_window: int = 16
    peak_window: int = 8
    peak_tol: float = 0.005
    volume_median_window: int = 20
    vol_ratio_spike: float = 5.0
    vol_fade_ratio: float = 0.85
    corridor_window: int = 30
    corridor_quantile: float = 0.95
    rsi_hot: float = 70.0
    mfi_hot: float = 82.5
    rsi_extreme: float = 85.0
    mfi_extreme: float = 85.0
    rsi_fade_ratio: float = 0.98
    macd_fade_ratio: float = 0.99
    wick_high: float = 0.28
    wick_low: float = 0.20
    close_pos_high: float = 0.60
    close_pos_low: float = 0.35
    wick_blowoff: float = 0.35
    body_blowoff: float = 0.25
    cooldown_bars: int = 4

    liquidity_window_days: int = 7
    liquidity_window_bars: int = 672
    eqh_min_touches: int = 2
    eqh_base_tol: float = 0.001
    eqh_atr_factor: float = 0.25


DEFAULT_PUMP_PARAMS = PumpParams()
