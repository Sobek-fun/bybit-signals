from __future__ import annotations

from datetime import UTC, datetime, timedelta

FIFTEEN_MINUTES = timedelta(minutes=15)
FIFTEEN_MINUTES_SECONDS = int(FIFTEEN_MINUTES.total_seconds())


def normalize_timestamp(value: datetime | str) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("timestamp must be datetime or ISO string")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_15m_grid(ts: datetime | str) -> bool:
    dt = normalize_timestamp(ts)
    return dt.second == 0 and dt.microsecond == 0 and dt.minute % 15 == 0


def context_to_decision_time(context_bar_open_time: datetime | str) -> datetime:
    context = normalize_timestamp(context_bar_open_time)
    if not is_15m_grid(context):
        raise ValueError("context_bar_open_time must be aligned to 15m grid")
    return context + FIFTEEN_MINUTES


def ceil_to_15m(ts: datetime | str) -> datetime:
    dt = normalize_timestamp(ts)
    epoch_seconds = dt.timestamp()
    ceiled = int(-(-epoch_seconds // FIFTEEN_MINUTES_SECONDS) * FIFTEEN_MINUTES_SECONDS)
    return datetime.fromtimestamp(ceiled, tz=UTC)


def decision_to_entry_bar_open_time(decision_time: datetime | str, entry_shift_bars: int = 0) -> datetime:
    if entry_shift_bars < 0:
        raise ValueError("entry_shift_bars must be non-negative")
    decision = normalize_timestamp(decision_time)
    entry_base = ceil_to_15m(decision)
    return entry_base + (FIFTEEN_MINUTES * entry_shift_bars)


def validate_causality(
    context_bar_open_time: datetime | str,
    decision_time: datetime | str,
    entry_bar_open_time: datetime | str,
) -> None:
    context = normalize_timestamp(context_bar_open_time)
    decision = normalize_timestamp(decision_time)
    entry = normalize_timestamp(entry_bar_open_time)
    if not is_15m_grid(context):
        raise ValueError("context_bar_open_time must be aligned to 15m grid")
    if not is_15m_grid(entry):
        raise ValueError("entry_bar_open_time must be aligned to 15m grid")
    min_decision = context + FIFTEEN_MINUTES
    if decision < min_decision:
        raise ValueError("decision_time must be after context bar close")
    if entry < ceil_to_15m(decision):
        raise ValueError("entry_bar_open_time must not be earlier than allowed entry grid")


def is_valid_ideal_entry(entry_bar_open_time: datetime | str) -> bool:
    return is_15m_grid(entry_bar_open_time)
