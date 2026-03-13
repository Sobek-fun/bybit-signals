from __future__ import annotations

from typing import Any

from scripts.runpod_jobs.models import RunStateName


def normalize_state(payload: dict[str, Any]) -> RunStateName:
    raw = str(payload.get("state", "")).upper()
    try:
        return RunStateName(raw)
    except Exception:
        return RunStateName.UNKNOWN


def can_transition(current: RunStateName, new: RunStateName) -> bool:
    allowed = {
        RunStateName.PREPARING: {RunStateName.DEPLOYING, RunStateName.FAILED},
        RunStateName.DEPLOYING: {RunStateName.BOOTSTRAPPING, RunStateName.FAILED},
        RunStateName.BOOTSTRAPPING: {RunStateName.RUNNING, RunStateName.FAILED},
        RunStateName.RUNNING: {RunStateName.FINISHED, RunStateName.FAILED},
        RunStateName.FINISHED: set(),
        RunStateName.FAILED: set(),
        RunStateName.UNKNOWN: {RunStateName.PREPARING, RunStateName.FAILED, RunStateName.UNKNOWN},
    }
    return new in allowed.get(current, set())
