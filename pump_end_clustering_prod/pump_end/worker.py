import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from pump_end_clustering_prod.pump_end.feature_builder import PumpFeatureBuilder
from pump_end_clustering_prod.pump_end.model import PumpEndClusteringModel
from pump_end_clustering_prod.pump_end.params import PumpParams, DEFAULT_PUMP_PARAMS
from pump_end_clustering_prod.delivery.signal_dispatcher import SignalDispatcher
from pump_end_clustering_prod.infra.logging import log


@dataclass
class PumpEndWorkerResult:
    token: str
    symbol: str
    status: str
    duration_total_ms: float
    duration_features_ms: float = 0
    duration_predict_ms: float = 0
    duration_base_indicators_ms: float = 0
    duration_pump_detector_ms: float = 0
    duration_liquidity_ms: float = 0
    duration_shift_ms: float = 0
    duration_extract_ms: float = 0
    candles_count: int = 0
    p_end: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_allowed: Optional[bool] = None
    signal_triggered: bool = False
    error_message: Optional[str] = None


@dataclass
class _EventState:
    event_id: str
    candidate_time: datetime
    cluster_id: Optional[int] = None
    params: Optional[dict] = None
    allowed: bool = True
    offsets_processed: list = field(default_factory=list)
    p_end_series: list = field(default_factory=list)
    signal_fired: bool = False


class PumpEndSignalState:
    def __init__(self):
        self.active_events: dict[str, _EventState] = {}
        self.processed_candidates: dict[str, set] = {}
        self.last_update_time: dict[str, datetime] = {}

    def get_processed_candidates(self, symbol: str) -> set:
        return self.processed_candidates.get(symbol, set())

    def register_event(
            self,
            symbol: str,
            event_id: str,
            candidate_time: datetime,
            cluster_id: Optional[int],
            params: Optional[dict],
            allowed: bool
    ):
        if symbol not in self.processed_candidates:
            self.processed_candidates[symbol] = set()
        self.processed_candidates[symbol].add(candidate_time)

        self.active_events[event_id] = _EventState(
            event_id=event_id,
            candidate_time=candidate_time,
            cluster_id=cluster_id,
            params=params,
            allowed=allowed,
        )

    def update_and_check(
            self,
            symbol: str,
            event_id: str,
            offset: int,
            p_end: float,
            current_time: datetime
    ) -> bool:
        last_t = self.last_update_time.get(symbol)
        if last_t is not None and current_time < last_t:
            return False
        self.last_update_time[symbol] = current_time

        ev = self.active_events.get(event_id)
        if ev is None or ev.signal_fired or not ev.allowed:
            return False

        params = ev.params
        if params is None:
            return False

        threshold = params['threshold']
        min_pending_bars = params['min_pending_bars']
        drop_delta = params['drop_delta']
        min_pending_peak = params['min_pending_peak']
        min_turn_down_bars = params['min_turn_down_bars']

        ev.offsets_processed.append(offset)
        ev.p_end_series.append(p_end)

        series = ev.p_end_series

        pending_count = 0
        pending_max = -np.inf
        turn_down_count = 0
        triggered = False

        for i in range(len(series)):
            if series[i] >= threshold:
                pending_count += 1
                pending_max = max(pending_max, series[i])

                if i > 0 and series[i] < series[i - 1]:
                    turn_down_count += 1
                else:
                    turn_down_count = 0

                if pending_count >= min_pending_bars and pending_max >= min_pending_peak and i > 0:
                    drop_from_peak = pending_max - series[i]
                    if drop_from_peak >= drop_delta and turn_down_count >= min_turn_down_bars:
                        triggered = True
                        break
            else:
                pending_count = 0
                pending_max = -np.inf
                turn_down_count = 0

        if triggered:
            ev.signal_fired = True
            return True

        return False

    def cleanup_finished_events(self, symbol: str, neg_after: int):
        to_remove = []
        for eid, ev in self.active_events.items():
            if not eid.startswith(f"{symbol}|"):
                continue
            if ev.signal_fired:
                to_remove.append(eid)
                continue
            if ev.offsets_processed and max(ev.offsets_processed) >= neg_after:
                to_remove.append(eid)
        for eid in to_remove:
            del self.active_events[eid]


def detect_candidates(df: pd.DataFrame, params: PumpParams) -> list:
    df = df.copy()

    df['vol_median'] = df['volume'].rolling(window=params.volume_median_window).median()
    df['vol_ratio'] = df['volume'] / df['vol_median']

    df.ta.rsi(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df = df.rename(columns={
        'RSI_14': 'rsi_14',
        'MFI_14': 'mfi_14',
        'MACDh_12_26_9': 'macdh_12_26_9',
        'MACD_12_26_9': 'macd_line',
        'MACDs_12_26_9': 'macd_signal'
    })

    rsi = df['rsi_14']
    mfi = df['mfi_14']
    macdh = df['macdh_12_26_9']
    macd_line = df['macd_line']

    df['vol_ratio_max'] = df['vol_ratio'].rolling(window=params.peak_window).max()
    df['rsi_max'] = rsi.rolling(window=params.peak_window).max()
    df['macdh_max'] = macdh.rolling(window=params.peak_window).max()
    df['high_max'] = df['high'].rolling(window=params.peak_window).max()

    min_price = df[['low', 'open']].min(axis=1)
    local_min = min_price.rolling(window=params.runup_window).min()
    runup = (df['high'] / local_min) - 1
    runup_met = ((local_min > 0) & (runup >= params.runup_threshold)).fillna(False)

    vol_spike_cond = df['vol_ratio'] >= params.vol_ratio_spike
    vol_spike_recent = vol_spike_cond.rolling(window=params.context_window).sum().fillna(0) > 0

    rsi_corridor = rsi.rolling(window=params.corridor_window).quantile(params.corridor_quantile)
    mfi_corridor = mfi.rolling(window=params.corridor_window).quantile(params.corridor_quantile)

    rsi_hot = rsi.notna() & rsi_corridor.notna() & (rsi >= np.maximum(params.rsi_hot, rsi_corridor))
    mfi_hot = mfi.notna() & mfi_corridor.notna() & (mfi >= np.maximum(params.mfi_hot, mfi_corridor))
    osc_hot_recent = (rsi_hot | mfi_hot).rolling(window=params.context_window).sum().fillna(0) > 0

    macd_pos_recent = (macdh.notna() & (macdh > 0)).rolling(window=params.context_window).sum().fillna(0) > 0

    pump_ctx = runup_met & vol_spike_recent & osc_hot_recent & macd_pos_recent

    high_max = df['high_max']
    near_peak = high_max.notna() & (high_max > 0) & (df['high'] >= high_max * (1 - params.peak_tol))

    n = len(df)
    open_p = df['open'].values
    high_p = df['high'].values
    low_p = df['low'].values
    close_p = df['close'].values

    candle_range_arr = high_p - low_p
    range_pos_arr = candle_range_arr > 0

    close_pos = np.zeros(n, dtype=float)
    np.divide(close_p - low_p, candle_range_arr, out=close_pos, where=range_pos_arr)

    max_oc = np.maximum(open_p, close_p)
    upper_wick = high_p - max_oc

    wick_ratio = np.zeros(n, dtype=float)
    np.divide(upper_wick, candle_range_arr, out=wick_ratio, where=range_pos_arr)

    body_size = np.abs(close_p - open_p)
    body_ratio = np.zeros(n, dtype=float)
    np.divide(body_size, candle_range_arr, out=body_ratio, where=range_pos_arr)

    bearish = close_p < open_p

    blowoff_exhaustion = (
            (close_pos <= params.close_pos_low) |
            (bearish & (close_pos <= 0.45)) |
            ((wick_ratio >= params.wick_blowoff) & (body_ratio <= params.body_blowoff))
    )

    osc_extreme = rsi.notna() & mfi.notna() & (rsi >= params.rsi_extreme) & (mfi >= params.mfi_extreme)
    predump_mask = osc_extreme & (close_pos >= params.close_pos_high)

    vol_ratio_max = df['vol_ratio_max']
    vol_ratio = df['vol_ratio']
    vol_fade = vol_ratio_max.notna() & (vol_ratio_max > 0) & (vol_ratio <= vol_ratio_max * params.vol_fade_ratio)

    wick_high_mask = wick_ratio >= params.wick_high
    wick_low_mask = (wick_ratio >= params.wick_low) & (~wick_high_mask)

    rsi_max = df['rsi_max']
    macdh_max = df['macdh_max']
    rsi_fade = rsi_max.notna() & (rsi_max > 0) & (rsi <= rsi_max * params.rsi_fade_ratio)
    macd_fade = macdh_max.notna() & macdh.notna() & (macdh_max > 0) & (macdh <= macdh_max * params.macd_fade_ratio)

    predump_peak = (
            predump_mask &
            (
                    (wick_high_mask & vol_fade) |
                    (wick_low_mask & vol_fade & (rsi_fade | macd_fade))
            )
    ).fillna(False)

    pump_ctx_arr = pump_ctx.to_numpy(dtype=bool, copy=False)
    near_peak_arr = near_peak.to_numpy(dtype=bool, copy=False)
    blowoff_arr = np.array(blowoff_exhaustion, dtype=bool)
    predump_arr = predump_peak.to_numpy(dtype=bool, copy=False)

    strong_cond = pump_ctx_arr & near_peak_arr & (blowoff_arr | predump_arr)

    macd_turn_down = (
            macd_line.notna() &
            macd_line.shift(1).notna() &
            (macd_line < macd_line.shift(1))
    ).to_numpy(dtype=bool, copy=False)

    candidates = []
    pending = False
    last_signal_idx = None

    skip_initial = max(
        params.volume_median_window,
        params.runup_window,
        params.corridor_window,
        params.context_window,
        params.peak_window
    )

    for i in range(skip_initial, n):
        if strong_cond[i] and not pending:
            pending = True

        if pending and macd_turn_down[i]:
            if last_signal_idx is None or i - last_signal_idx >= params.cooldown_bars:
                candidates.append(df.index[i])
                last_signal_idx = i
            pending = False

    return candidates


def build_points_around_candidates(
        candidates: list,
        symbol: str,
        neg_before: int,
        neg_after: int,
        pos_offsets: list
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()

    all_offsets = list(range(-neg_before, 0)) + pos_offsets + list(
        range(max(pos_offsets) + 1, max(pos_offsets) + neg_after + 1)
    )
    all_offsets = sorted(set(all_offsets))

    rows = []
    for event_time in candidates:
        event_id = f"{symbol}|{event_time.strftime('%Y%m%d_%H%M%S')}"
        for offset in all_offsets:
            open_time = event_time + timedelta(minutes=offset * 15)
            rows.append({
                'event_id': event_id,
                'symbol': symbol,
                'open_time': open_time,
                'offset': offset,
                'y': 1 if offset in pos_offsets else 0,
                'pump_la_type': 'A',
                'runup_pct': 0
            })

    return pd.DataFrame(rows)


class PumpEndWorker:
    def __init__(
            self,
            token: str,
            df: pd.DataFrame,
            expected_bucket_start: datetime,
            model: PumpEndClusteringModel,
            feature_builder: PumpFeatureBuilder,
            signal_state: PumpEndSignalState,
            signal_dispatcher: SignalDispatcher,
            min_candles: int
    ):
        self.token = token
        self.symbol = f"{token}USDT"
        self.df = df
        self.expected_bucket_start = expected_bucket_start
        self.model = model
        self.feature_builder = feature_builder
        self.signal_state = signal_state
        self.signal_dispatcher = signal_dispatcher
        self.min_candles = min_candles

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

            if len(self.df) < self.min_candles:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SKIP_GAPPED",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    candles_count=len(self.df)
                )

            decision_open_time = self.expected_bucket_start + timedelta(minutes=15)

            candidates = detect_candidates(self.df, DEFAULT_PUMP_PARAMS)

            already_processed = self.signal_state.get_processed_candidates(self.symbol)
            new_candidates = [c for c in candidates if c not in already_processed]

            neg_before = self.model.neg_before
            neg_after = self.model.neg_after
            pos_offsets = self.model.pos_offsets
            max_offset = max(pos_offsets) + neg_after

            fresh_candidates = []
            for c in new_candidates:
                offset = int((decision_open_time - c) / timedelta(minutes=15))
                if offset > max_offset:
                    if self.symbol not in self.signal_state.processed_candidates:
                        self.signal_state.processed_candidates[self.symbol] = set()
                    self.signal_state.processed_candidates[self.symbol].add(c)
                else:
                    fresh_candidates.append(c)
            new_candidates = fresh_candidates

            if not new_candidates:
                active_event_ids = [
                    eid for eid in self.signal_state.active_events
                    if eid.startswith(f"{self.symbol}|")
                ]
                if not active_event_ids:
                    return PumpEndWorkerResult(
                        token=self.token,
                        symbol=self.symbol,
                        status="OK_NO_SIGNAL",
                        duration_total_ms=(time.time() - start_time) * 1000,
                        candles_count=len(self.df)
                    )

            for cand_time in new_candidates:
                event_id = f"{self.symbol}|{cand_time.strftime('%Y%m%d_%H%M%S')}"

                points_df = build_points_around_candidates(
                    [cand_time], self.symbol, neg_before, neg_after, pos_offsets
                )
                if points_df.empty:
                    continue

                offset_zero_rows = points_df[points_df['offset'] == 0]
                if offset_zero_rows.empty:
                    continue

                zero_time = offset_zero_rows.iloc[0]['open_time']

                features_start = time.time()
                zero_features, _ = self.feature_builder.build_one_for_inference(
                    self.df, self.symbol, zero_time
                )
                if not zero_features:
                    continue

                cluster_id, params, allowed = self.model.resolve_params(zero_features)

                if not allowed:
                    if self.symbol not in self.signal_state.processed_candidates:
                        self.signal_state.processed_candidates[self.symbol] = set()
                    self.signal_state.processed_candidates[self.symbol].add(cand_time)
                    continue

                self.signal_state.register_event(
                    self.symbol, event_id, cand_time, cluster_id, params, allowed
                )

                current_offset = int((decision_open_time - cand_time) / timedelta(minutes=15))
                backfill_offsets = []
                backfill_times = []
                for o in range(-neg_before, current_offset):
                    t = cand_time + timedelta(minutes=o * 15)
                    if t in self.df.index:
                        backfill_offsets.append(o)
                        backfill_times.append(t)

                if backfill_times:
                    backfill_features = self.feature_builder.build_many_for_inference(
                        self.df, self.symbol, backfill_times
                    )
                    if backfill_features and len(backfill_features) == len(backfill_times):
                        backfill_probas = self.model.batch_predict(backfill_features)
                        ev = self.signal_state.active_events[event_id]
                        for i, o in enumerate(backfill_offsets):
                            ev.offsets_processed.append(o)
                            ev.p_end_series.append(float(backfill_probas[i]))

            triggered = False
            last_p_end = None
            last_cluster_id = None
            last_allowed = None
            last_params = None
            total_features_ms = 0
            total_predict_ms = 0
            timings_agg = {}

            active_event_ids = [
                eid for eid in self.signal_state.active_events
                if eid.startswith(f"{self.symbol}|")
            ]

            for event_id in active_event_ids:
                ev = self.signal_state.active_events.get(event_id)
                if ev is None or ev.signal_fired:
                    continue

                current_offset_time = decision_open_time
                candidate_time = ev.candidate_time
                offset = int((current_offset_time - candidate_time) / timedelta(minutes=15))

                min_offset = -neg_before
                if offset < min_offset or offset > max_offset:
                    continue

                if offset in ev.offsets_processed:
                    continue

                feat_start = time.time()
                features_row, timings = self.feature_builder.build_one_for_inference(
                    self.df, self.symbol, decision_open_time
                )
                feat_ms = (time.time() - feat_start) * 1000
                total_features_ms += feat_ms
                if not timings_agg:
                    timings_agg = timings

                if not features_row:
                    continue

                pred_start = time.time()
                p_end = self.model.predict(features_row)
                pred_ms = (time.time() - pred_start) * 1000
                total_predict_ms += pred_ms

                last_p_end = p_end
                last_cluster_id = ev.cluster_id
                last_allowed = ev.allowed
                last_params = ev.params

                sig = self.signal_state.update_and_check(
                    self.symbol, event_id, offset, p_end, decision_open_time
                )

                if sig:
                    triggered = True
                    last_close = self.df.loc[self.expected_bucket_start, 'close']

                    self.signal_dispatcher.publish_pump_end_signal(
                        symbol=self.symbol,
                        event_time=decision_open_time,
                        p_end=p_end,
                        threshold=ev.params['threshold'],
                        close_price=last_close,
                        min_pending_bars=ev.params['min_pending_bars'],
                        drop_delta=ev.params['drop_delta'],
                        min_pending_peak=ev.params['min_pending_peak'],
                        min_turn_down_bars=ev.params['min_turn_down_bars'],
                        cluster_id=ev.cluster_id
                    )
                    break

            self.signal_state.cleanup_finished_events(self.symbol, neg_after)

            if triggered:
                return PumpEndWorkerResult(
                    token=self.token,
                    symbol=self.symbol,
                    status="SIGNAL_SENT",
                    duration_total_ms=(time.time() - start_time) * 1000,
                    duration_features_ms=total_features_ms,
                    duration_predict_ms=total_predict_ms,
                    duration_base_indicators_ms=timings_agg.get('base_indicators_ms', 0),
                    duration_pump_detector_ms=timings_agg.get('pump_detector_ms', 0),
                    duration_liquidity_ms=timings_agg.get('liquidity_ms', 0),
                    duration_shift_ms=timings_agg.get('shift_ms', 0),
                    duration_extract_ms=timings_agg.get('extract_ms', 0),
                    candles_count=len(self.df),
                    p_end=last_p_end,
                    cluster_id=last_cluster_id,
                    cluster_allowed=last_allowed,
                    signal_triggered=True
                )

            if last_allowed is not None and not last_allowed:
                status = "OK_CLUSTER_BLOCKED"
            else:
                status = "OK_NO_SIGNAL"

            return PumpEndWorkerResult(
                token=self.token,
                symbol=self.symbol,
                status=status,
                duration_total_ms=(time.time() - start_time) * 1000,
                duration_features_ms=total_features_ms,
                duration_predict_ms=total_predict_ms,
                duration_base_indicators_ms=timings_agg.get('base_indicators_ms', 0),
                duration_pump_detector_ms=timings_agg.get('pump_detector_ms', 0),
                duration_liquidity_ms=timings_agg.get('liquidity_ms', 0),
                duration_shift_ms=timings_agg.get('shift_ms', 0),
                duration_extract_ms=timings_agg.get('extract_ms', 0),
                candles_count=len(self.df),
                p_end=last_p_end,
                cluster_id=last_cluster_id,
                cluster_allowed=last_allowed,
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
