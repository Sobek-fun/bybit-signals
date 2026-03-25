from __future__ import annotations

from datetime import datetime

ALLOWED_COMPONENTS = {
    "RUN",
    "CONFIG",
    "ARTIFACTS",
    "DRYRUN",
    "TEST",
    "EVENT",
    "RESOLVER",
    "ROWS",
    "METRICS",
    "LAYERS",
    "FEATURES",
    "DETECTOR",
    "CV",
    "POLICY",
    "GATE",
    "EXECUTION",
    "IO",
    "PIPELINE",
}


def _log(level: str, component: str, message: str) -> None:
    if component not in ALLOWED_COMPONENTS:
        raise ValueError(f"unsupported component: {component}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} [{component}] {message}")


def log_info(component: str, message: str) -> None:
    _log("INFO", component, message)


def log_warn(component: str, message: str) -> None:
    _log("WARN", component, message)


def log_error(component: str, message: str) -> None:
    _log("ERROR", component, message)


def stage_start(component: str, stage: str) -> None:
    _log("INFO", component, f"{stage} start")


def stage_done(component: str, stage: str, elapsed_sec: float | None = None) -> None:
    if elapsed_sec is None:
        _log("INFO", component, f"{stage} done")
        return
    _log("INFO", component, f"{stage} done elapsed_sec={elapsed_sec:.3f}")


def progress(component: str, message: str) -> None:
    _log("INFO", component, message)
