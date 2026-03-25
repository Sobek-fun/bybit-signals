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
class EventOpenerConfig:
    runup_lookback_bars: int
    min_runup_pct: float
    near_high_lookback_bars: int
    near_high_tol_pct: float
    volume_ratio_lookback_bars: int
    min_volume_ratio: float
    max_episode_bars: int
    expiry_drawdown_pct: float
    cooldown_bars: int


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    horizon_bars: int
    success_pullback_pct: float
    max_prepullback_squeeze_pct: float
    flat_max_abs_move_pct: float


@dataclass(frozen=True, slots=True)
class V2Config:
    raw: dict[str, Any]
    splits: SplitBounds
    references: ReferenceSymbolsConfig
    event_opener: EventOpenerConfig
    resolver: ResolverConfig
    execution: ExecutionContract


@dataclass(frozen=True, slots=True)
class ReferenceSymbolsConfig:
    btc_symbol: str
    eth_symbol: str


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
    references = _build_references(config["data"]["references"])
    event_opener = _build_event_opener(config["data"]["event_opener"])
    resolver = _build_resolver(config["data"]["resolver"])
    execution = ExecutionContract(
        tp_pct=float(config["execution"]["tp_pct"]),
        sl_pct=float(config["execution"]["sl_pct"]),
        max_hold_bars=int(config["execution"]["max_hold_bars"]),
        entry_shift_bars=int(config["execution"]["entry_shift_bars"]),
        replay_resolution=str(config["execution"]["replay_resolution"]),
    )
    return V2Config(
        raw=copy.deepcopy(config),
        splits=splits,
        references=references,
        event_opener=event_opener,
        resolver=resolver,
        execution=execution,
    )


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
    event_opener_section = data_section.get("event_opener")
    if not isinstance(event_opener_section, dict):
        raise ValueError("missing required section: data.event_opener")
    resolver_section = data_section.get("resolver")
    if not isinstance(resolver_section, dict):
        raise ValueError("missing required section: data.resolver")
    references_section = data_section.get("references")
    if not isinstance(references_section, dict):
        raise ValueError("missing required section: data.references")
    _validate_event_opener(event_opener_section)
    _validate_resolver(resolver_section)
    _validate_references(references_section)


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


def _require_positive_int(value: Any, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_non_negative_int(value: Any, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _require_positive_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_non_negative_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _validate_event_opener(section: dict[str, Any]) -> None:
    _build_event_opener(section)


def _validate_resolver(section: dict[str, Any]) -> None:
    _build_resolver(section)


def _build_event_opener(section: dict[str, Any]) -> EventOpenerConfig:
    return EventOpenerConfig(
        runup_lookback_bars=_require_positive_int(
            section.get("runup_lookback_bars"), "data.event_opener.runup_lookback_bars"
        ),
        min_runup_pct=_require_positive_float(section.get("min_runup_pct"), "data.event_opener.min_runup_pct"),
        near_high_lookback_bars=_require_positive_int(
            section.get("near_high_lookback_bars"), "data.event_opener.near_high_lookback_bars"
        ),
        near_high_tol_pct=_require_non_negative_float(
            section.get("near_high_tol_pct"), "data.event_opener.near_high_tol_pct"
        ),
        volume_ratio_lookback_bars=_require_positive_int(
            section.get("volume_ratio_lookback_bars"), "data.event_opener.volume_ratio_lookback_bars"
        ),
        min_volume_ratio=_require_positive_float(
            section.get("min_volume_ratio"), "data.event_opener.min_volume_ratio"
        ),
        max_episode_bars=_require_positive_int(section.get("max_episode_bars"), "data.event_opener.max_episode_bars"),
        expiry_drawdown_pct=_require_positive_float(
            section.get("expiry_drawdown_pct"), "data.event_opener.expiry_drawdown_pct"
        ),
        cooldown_bars=_require_non_negative_int(section.get("cooldown_bars"), "data.event_opener.cooldown_bars"),
    )


def _build_resolver(section: dict[str, Any]) -> ResolverConfig:
    return ResolverConfig(
        horizon_bars=_require_positive_int(section.get("horizon_bars"), "data.resolver.horizon_bars"),
        success_pullback_pct=_require_positive_float(
            section.get("success_pullback_pct"), "data.resolver.success_pullback_pct"
        ),
        max_prepullback_squeeze_pct=_require_positive_float(
            section.get("max_prepullback_squeeze_pct"), "data.resolver.max_prepullback_squeeze_pct"
        ),
        flat_max_abs_move_pct=_require_positive_float(
            section.get("flat_max_abs_move_pct"), "data.resolver.flat_max_abs_move_pct"
        ),
    )


def _validate_references(section: dict[str, Any]) -> None:
    _build_references(section)


def _build_references(section: dict[str, Any]) -> ReferenceSymbolsConfig:
    btc_symbol = section.get("btc_symbol")
    if not isinstance(btc_symbol, str) or not btc_symbol.strip():
        raise ValueError("data.references.btc_symbol must be a non-empty string")
    eth_symbol = section.get("eth_symbol")
    if not isinstance(eth_symbol, str) or not eth_symbol.strip():
        raise ValueError("data.references.eth_symbol must be a non-empty string")
    return ReferenceSymbolsConfig(
        btc_symbol=btc_symbol.strip(),
        eth_symbol=eth_symbol.strip(),
    )


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
