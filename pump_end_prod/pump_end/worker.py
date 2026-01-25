import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from pump_end_prod.pump_end.feature_builder import PumpFeatureBuilder
from pump_end_prod.pump_end.model import PumpEndModel
from pump_end_prod.delivery.signal_dispatcher import SignalDispatcher


@dataclass
class PumpEndWorkerResult:
    token: str
    symbol: str
    status: str
    duration_total_ms: float
    duration_features_ms: float = 0
    duration_predict_ms: float = 0
    candles_count: int = 0
    p_end: Optional[float] = None
    signal_triggered: bool = False
    error_message: Optional[str] = None


class PumpEndSignalState:
    def __init__(self):
        self.prev_p_end: dict[str, float] = {}
        self.pending_count: dict[str, int] = {}
        self.last_signal_time: dict[str, datetime] = {}
        self.disarmed: dict[str, bool] = {}

    def update_and_check(
            self,
            symbol: str,
            p_end: float,
            threshold: float,
            min_pending_bars: int,
            drop_delta: float,
            current_time: datetime
    ) -> bool:
        prev = self.prev_p_end.get(symbol)
        count = self.pending_count.get(symbol, 0)
        is_disarmed = self.disarmed.get(symbol, False)

        triggered = False

        if p_end >= threshold:
            if is_disarmed:
                pass
            else:
                count += 1
                if count >= min_pending_bars and prev is not None:
                    drop = prev - p_end
                    if p_end < prev and drop >= drop_delta:
                        if self.last_signal_time.get(symbol) != current_time:
                            triggered = True
                            self.last_signal_time[symbol] = current_time
                            self.disarmed[symbol] = True
        else:
            count = 0
            self.disarmed[symbol] = False

        self.pending_count[symbol] = count
        self.prev_p_end[symbol] = p_end

        return triggered


class PumpEndWorker:
    MIN_CANDLES = 873

    def __init__(
            self,
            token: str,
            df: pd.DataFrame,
            expected_bucket_start: datetime,
            model: PumpEndModel,
            feature_builder: PumpFeatureBuilder,
            signal_state: PumpEndSignalState,
            signal_dispatcher: SignalDispatcher
    ):
        self.token = token
        self.symbol = f"{token}USDT"
        self.df = df
        self.expected_bucket_start = expected_bucket_start
        self.model = model
        self.feature_builder = feature_builder
        self.signal_state = signal_state
        self.signal_dispatcher = signal_dispatcher

    def process(self) -> PumpEndWorkerResult:
        start_time = time.time()

        try:
            if self.df.empty:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_EMPTY",
                    duration_total_ms=(time.time() - start_time) * 1000
                )

            if self.expected_bucket_start not in self.df.index:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_MISSING_LAST",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    candles_count=len(self.df)
                )

            last_candle_time = self.df.index[-1]
            if last_candle_time != self.expected_bucket_start:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_MISSING_LAST",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    candles_count=len(self.df)
                )

            if len(self.df) != self.MIN_CANDLES:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_GAPPED",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    candles_count=len(self.df)
                )

            start_bucket = self.expected_bucket_start - timedelta(minutes=(self.MIN_CANDLES - 1) * 15)
            if self.df.index[0] != start_bucket:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_GAPPED",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    candles_count=len(self.df)
                )

            decision_open_time = self.expected_bucket_start + timedelta(minutes=15)

            features_start = time.time()
            features_row = self.feature_builder.build_one_for_inference(
                self.df,
                self.symbol,
                decision_open_time
            )
            features_duration = (time.time() - features_start) * 1000

            if not features_row:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_FEATURES_EMPTY",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    duration_features_ms=features_duration,
                    candles_count=len(self.df)
                )

            predict_start = time.time()
            p_end = self.model.predict(features_row)
            predict_duration = (time.time() - predict_start) * 1000

            triggered = self.signal_state.update_and_check(
                self.symbol,
                p_end,
                self.model.threshold,
                self.model.min_pending_bars,
                self.model.drop_delta,
                decision_open_time
            )

            if triggered:
                last_close = self.df.loc[self.expected_bucket_start, 'close']

                self.signal_dispatcher.publish_pump_end_signal(
                    symbol=self.symbol,
                    event_time=decision_open_time,
                    p_end=p_end,
                    threshold=self.model.threshold,
                    close_price=last_close,
                    min_pending_bars=self.model.min_pending_bars,
                    drop_delta=self.model.drop_delta
                )

                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SIGNAL_SENT",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    duration_features_ms=features_duration,
                    duration_predict_ms=predict_duration,
                    candles_count=len(self.df),
                    p_end=p_end,
                    signal_triggered=True
                )

            return PumpEndWorkerResult(
                token=self.token,
                symbol=self.symbol,
                status="OK_NO_SIGNAL",
                duration_total_ms=(time.time() - start_time) * 1000,
                duration_features_ms=features_duration,
                duration_predict_ms=predict_duration,
                candles_count=len(self.df),
                p_end=p_end,
                signal_triggered=False
            )

        except Exception as e:
            return PumpEndWorkerResult(
                token=self.token,
                symbol=self.symbol,
                status="ERROR",
                duration_total_ms=(time.time() - start_time) * 1000,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
