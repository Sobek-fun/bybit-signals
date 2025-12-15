from dataclasses import dataclass


@dataclass
class Config:
    tokens: list[str]
    ch_dsn: str
    bot_token: str
    chat_id: str
    workers: int = 8
    offset_seconds: int = 10
    lookback_candles: int = 150
