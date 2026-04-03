from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pump_end_v2.time_utils import (
    decision_to_entry_bar_open_time,
    is_15m_grid,
    validate_causality,
)


class OutcomeClass(StrEnum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    FLAT = "flat"


class SignalQualityClass(StrEnum):
    CLEAN_RETRACE_H32 = "clean_retrace_h32"
    DIRTY_RETRACE_H32 = "dirty_retrace_h32"
    CLEAN_NO_PULLBACK_H32 = "clean_no_pullback_h32"
    DIRTY_NO_PULLBACK_H32 = "dirty_no_pullback_h32"
    PULLBACK_BEFORE_SQUEEZE_H32 = "pullback_before_squeeze_h32"


class TargetReason(StrEnum):
    TP = "tp"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"
    SL = "sl"
    TIMEOUT = "timeout"
    AMBIGUOUS = "ambiguous"
    INVALID_CONTEXT = "invalid_context"


class TradeOutcome(StrEnum):
    TP = "tp"
    SL = "sl"
    TIMEOUT = "timeout"
    AMBIGUOUS = "ambiguous"


class GateDecision(StrEnum):
    KEEP = "keep"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    episode_id: str
    symbol: str
    episode_open_time: datetime
    status: str

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not self.status:
            raise ValueError("status must be non-empty")


@dataclass(frozen=True, slots=True)
class DecisionRowRef:
    decision_row_id: str
    episode_id: str
    symbol: str
    context_bar_open_time: datetime
    decision_time: datetime
    entry_bar_open_time: datetime

    def __post_init__(self) -> None:
        if not self.decision_row_id:
            raise ValueError("decision_row_id must be non-empty")
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        validate_causality(
            context_bar_open_time=self.context_bar_open_time,
            decision_time=self.decision_time,
            entry_bar_open_time=self.entry_bar_open_time,
        )


@dataclass(frozen=True, slots=True)
class CandidateSignalRef:
    signal_id: str
    episode_id: str
    symbol: str
    fire_decision_time: datetime
    entry_bar_open_time: datetime

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not is_15m_grid(self.entry_bar_open_time):
            raise ValueError("entry_bar_open_time must be aligned to 15m grid")
        min_entry = decision_to_entry_bar_open_time(
            self.fire_decision_time, entry_shift_bars=0
        )
        if self.entry_bar_open_time < min_entry:
            raise ValueError(
                "entry_bar_open_time must not be earlier than allowed entry grid"
            )


@dataclass(frozen=True, slots=True)
class ExecutedSignalRef:
    signal_id: str
    symbol: str
    entry_bar_open_time: datetime
    exit_time: datetime
    trade_outcome: TradeOutcome

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not is_15m_grid(self.entry_bar_open_time):
            raise ValueError("entry_bar_open_time must be aligned to 15m grid")
        if self.exit_time < self.entry_bar_open_time:
            raise ValueError("exit_time must not be earlier than entry_bar_open_time")


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    tp_pct: float
    sl_pct: float
    max_hold_bars: int
    entry_shift_bars: int

    def __post_init__(self) -> None:
        if self.tp_pct <= 0:
            raise ValueError("tp_pct must be positive")
        if self.sl_pct <= 0:
            raise ValueError("sl_pct must be positive")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")
        if self.entry_shift_bars < 0:
            raise ValueError("entry_shift_bars must be non-negative")
