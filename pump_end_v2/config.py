from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pump_end_v2.contracts import ExecutionContract

REQUIRED_SECTIONS = ("data", "splits", "detector", "gate", "execution", "compute")


@dataclass(frozen=True, slots=True)
class SplitBounds:
    train_end: datetime
    val_end: datetime
    test_end: datetime


@dataclass(frozen=True, slots=True)
class V2Config:
    raw: dict[str, Any]
    splits: SplitBounds
    execution: ExecutionContract


def load_toml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError("config root must be a table")
    return data


def validate_config(config: dict[str, Any]) -> V2Config:
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"missing required section: {section}")
    _reject_feature_flags(config)
    _validate_splits(config["splits"])
    _validate_data(config["data"])
    _validate_non_negative(config["detector"], "detector")
    _validate_non_negative(config["gate"], "gate")
    _validate_execution(config["execution"])
    _validate_compute(config["compute"])
    splits = _build_splits(config["splits"])
    execution = ExecutionContract(
        tp_pct=float(config["execution"]["tp_pct"]),
        sl_pct=float(config["execution"]["sl_pct"]),
        max_hold_bars=int(config["execution"]["max_hold_bars"]),
        entry_shift_bars=int(config["execution"]["entry_shift_bars"]),
        replay_resolution=str(config["execution"]["replay_resolution"]),
    )
    return V2Config(raw=copy.deepcopy(config), splits=splits, execution=execution)


def load_and_validate_config(path: str | Path) -> V2Config:
    config = load_toml_config(path)
    return validate_config(config)


def _parse_date(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be YYYY-MM-DD string")
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD string") from exc


def _build_splits(splits_section: dict[str, Any]) -> SplitBounds:
    return SplitBounds(
        train_end=_parse_date(splits_section.get("train_end"), "splits.train_end"),
        val_end=_parse_date(splits_section.get("val_end"), "splits.val_end"),
        test_end=_parse_date(splits_section.get("test_end"), "splits.test_end"),
    )


def _validate_splits(splits_section: dict[str, Any]) -> None:
    bounds = _build_splits(splits_section)
    if not (bounds.train_end < bounds.val_end < bounds.test_end):
        raise ValueError("split chronology must satisfy train_end < val_end < test_end")


def _validate_data(data_section: dict[str, Any]) -> None:
    required_paths = ["source_root"]
    for key in required_paths:
        path_value = data_section.get(key)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"data.{key} must be a non-empty path")
        if not Path(path_value).exists():
            raise ValueError(f"data.{key} path does not exist: {path_value}")


def _validate_non_negative(section: dict[str, Any], section_name: str) -> None:
    for key, value in section.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"{section_name}.{key} must be non-negative")


def _validate_execution(execution_section: dict[str, Any]) -> None:
    for key in ("tp_pct", "sl_pct"):
        if float(execution_section.get(key, 0)) <= 0:
            raise ValueError(f"execution.{key} must be positive")
    if int(execution_section.get("max_hold_bars", 0)) <= 0:
        raise ValueError("execution.max_hold_bars must be positive")
    if int(execution_section.get("entry_shift_bars", -1)) < 0:
        raise ValueError("execution.entry_shift_bars must be non-negative")


def _validate_compute(compute_section: dict[str, Any]) -> None:
    runs_root = compute_section.get("runs_root")
    if not isinstance(runs_root, str) or not runs_root:
        raise ValueError("compute.runs_root must be a non-empty path")
    max_workers = int(compute_section.get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("compute.max_workers must be >= 1")


def _reject_feature_flags(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in {"feature_flags", "feature_flag", "flags"}:
                raise ValueError(f"feature flags are forbidden: {path}.{key}")
            _reject_feature_flags(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            _reject_feature_flags(nested, f"{path}[{idx}]")
