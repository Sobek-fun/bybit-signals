from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Config:
    tokens: list[str]
    ch_dsn: str
    bot_token: str
    chat_id: str
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    workers: int = 8
    offset_seconds: int = 1
    lookback_candles: int = 150
    test_days: int = 30


@dataclass
class WorkerResult:
    token: str
    symbol: str
    status: str
    duration_total_ms: float
    duration_load_ms: float = 0
    duration_indicators_ms: float = 0
    duration_detect_ms: float = 0
    duration_telegram_ms: float = 0
    candles_count: int = 0
    last_bucket: Optional[datetime] = None
    error_stage: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
