from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    max_wait_bars_for_success: int


@dataclass(frozen=True, slots=True)
class DetectorModelConfig:
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_seed: int


@dataclass(frozen=True, slots=True)
class DetectorCVConfig:
    min_train_days: int
    fold_span_days: int
    max_folds: int


@dataclass(frozen=True, slots=True)
class DetectorPolicyConfig:
    arm_score_min: float
    fire_score_floor: float
    turn_down_delta: float


@dataclass(frozen=True, slots=True)
class GateConfig:
    block_threshold: float


@dataclass(frozen=True, slots=True)
class GateModelConfig:
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_seed: int


@dataclass(frozen=True, slots=True)
class ReferenceSymbolsConfig:
    btc_symbol: str
    eth_symbol: str


@dataclass(frozen=True, slots=True)
class DataFilesConfig:
    bars_15m_path: str
    bars_1m_path: str
    bars_1s_path: str


@dataclass(frozen=True, slots=True)
class V2Config:
    raw: dict[str, Any]
    splits: SplitBounds
    references: ReferenceSymbolsConfig
    data_files: DataFilesConfig
    event_opener: EventOpenerConfig
    resolver: ResolverConfig
    detector_model: DetectorModelConfig
    detector_cv: DetectorCVConfig
    detector_policy: DetectorPolicyConfig
    gate_config: GateConfig
    gate_model: GateModelConfig
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
    _validate_splits(config["splits"])
    _validate_data(config["data"])
    _validate_detector(config["detector"])
    _validate_gate(config["gate"])
    _validate_execution(config["execution"])
    _validate_compute(config["compute"])
    splits = _build_splits(config["splits"])
    references = _build_references(config["data"]["references"])
    data_files = _build_data_files(config["data"]["files"])
    event_opener = _build_event_opener(config["data"]["event_opener"])
    resolver = _build_resolver(config["data"]["resolver"])
    detector_model = _build_detector_model(config["detector"])
    detector_cv = _build_detector_cv(config["detector"])
    detector_policy = _build_detector_policy(config["detector"])
    gate_config = _build_gate_config(config["gate"])
    gate_model = _build_gate_model(config["gate"])
    execution = ExecutionContract(
        tp_pct=float(config["execution"]["tp_pct"]),
        sl_pct=float(config["execution"]["sl_pct"]),
        max_hold_bars=int(config["execution"]["max_hold_bars"]),
        entry_shift_bars=int(config["execution"]["entry_shift_bars"]),
    )
    return V2Config(
        raw=copy.deepcopy(config),
        splits=splits,
        references=references,
        data_files=data_files,
        event_opener=event_opener,
        resolver=resolver,
        detector_model=detector_model,
        detector_cv=detector_cv,
        detector_policy=detector_policy,
        gate_config=gate_config,
        gate_model=gate_model,
        execution=execution,
    )


def load_and_validate_config(path: str | Path) -> V2Config:
    config = load_toml_config(path)
    return validate_config(config)


def _parse_date(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be YYYY-MM-DD string")
    try:
        day_start = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD string") from exc
    return day_start + timedelta(days=1) - timedelta(microseconds=1)


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
    files_section = data_section.get("files")
    if not isinstance(files_section, dict):
        raise ValueError("missing required section: data.files")
    _validate_event_opener(event_opener_section)
    _validate_resolver(resolver_section)
    _validate_references(references_section)
    _validate_data_files(files_section)


def _validate_detector(detector_section: dict[str, Any]) -> None:
    _validate_non_negative(detector_section, "detector")
    _build_detector_model(detector_section)
    _build_detector_cv(detector_section)
    _build_detector_policy(detector_section)


def _validate_gate(gate_section: dict[str, Any]) -> None:
    _build_gate_config(gate_section)
    _build_gate_model(gate_section)


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


def _require_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be non-negative") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _require_positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_non_negative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be non-negative") from exc
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
    horizon_bars = _require_positive_int(section.get("horizon_bars"), "data.resolver.horizon_bars")
    max_wait_bars_for_success = _require_positive_int(
        section.get("max_wait_bars_for_success"),
        "data.resolver.max_wait_bars_for_success",
    )
    if max_wait_bars_for_success > horizon_bars:
        raise ValueError("data.resolver.max_wait_bars_for_success must be <= data.resolver.horizon_bars")
    return ResolverConfig(
        horizon_bars=horizon_bars,
        success_pullback_pct=_require_positive_float(
            section.get("success_pullback_pct"), "data.resolver.success_pullback_pct"
        ),
        max_prepullback_squeeze_pct=_require_positive_float(
            section.get("max_prepullback_squeeze_pct"), "data.resolver.max_prepullback_squeeze_pct"
        ),
        flat_max_abs_move_pct=_require_positive_float(
            section.get("flat_max_abs_move_pct"), "data.resolver.flat_max_abs_move_pct"
        ),
        max_wait_bars_for_success=max_wait_bars_for_success,
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
    if btc_symbol.strip() == eth_symbol.strip():
        raise ValueError("data.references.btc_symbol must differ from data.references.eth_symbol")
    return ReferenceSymbolsConfig(
        btc_symbol=btc_symbol.strip(),
        eth_symbol=eth_symbol.strip(),
    )


def _validate_data_files(section: dict[str, Any]) -> None:
    _build_data_files(section)


def _build_data_files(section: dict[str, Any]) -> DataFilesConfig:
    bars_15m_path = section.get("bars_15m_path")
    if not isinstance(bars_15m_path, str) or not bars_15m_path.strip():
        raise ValueError("data.files.bars_15m_path must be a non-empty string")
    bars_1m_path = section.get("bars_1m_path")
    if not isinstance(bars_1m_path, str) or not bars_1m_path.strip():
        raise ValueError("data.files.bars_1m_path must be a non-empty string")
    bars_1s_path = section.get("bars_1s_path")
    if not isinstance(bars_1s_path, str):
        raise ValueError("data.files.bars_1s_path must be a string")
    return DataFilesConfig(
        bars_15m_path=bars_15m_path.strip(),
        bars_1m_path=bars_1m_path.strip(),
        bars_1s_path=bars_1s_path.strip(),
    )


def _build_detector_model(section: dict[str, Any]) -> DetectorModelConfig:
    model_section = section.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("missing required section: detector.model")
    return DetectorModelConfig(
        iterations=_require_positive_int(model_section.get("iterations"), "detector.model.iterations"),
        depth=_require_positive_int(model_section.get("depth"), "detector.model.depth"),
        learning_rate=_require_positive_float(model_section.get("learning_rate"), "detector.model.learning_rate"),
        l2_leaf_reg=_require_positive_float(model_section.get("l2_leaf_reg"), "detector.model.l2_leaf_reg"),
        random_seed=_require_non_negative_int(model_section.get("random_seed"), "detector.model.random_seed"),
    )


def _build_detector_cv(section: dict[str, Any]) -> DetectorCVConfig:
    cv_section = section.get("cv")
    if not isinstance(cv_section, dict):
        raise ValueError("missing required section: detector.cv")
    return DetectorCVConfig(
        min_train_days=_require_positive_int(cv_section.get("min_train_days"), "detector.cv.min_train_days"),
        fold_span_days=_require_positive_int(cv_section.get("fold_span_days"), "detector.cv.fold_span_days"),
        max_folds=_require_positive_int(cv_section.get("max_folds"), "detector.cv.max_folds"),
    )


def _build_detector_policy(section: dict[str, Any]) -> DetectorPolicyConfig:
    arm_score_min = float(section.get("arm_score_min"))
    fire_score_floor = float(section.get("fire_score_floor"))
    turn_down_delta = float(section.get("turn_down_delta"))
    if not (0.0 < arm_score_min <= 1.0):
        raise ValueError("detector.arm_score_min must satisfy 0 < x <= 1")
    if not (0.0 <= fire_score_floor <= arm_score_min):
        raise ValueError("detector.fire_score_floor must satisfy 0 <= x <= detector.arm_score_min")
    if not (0.0 < turn_down_delta <= 1.0):
        raise ValueError("detector.turn_down_delta must satisfy 0 < x <= 1")
    return DetectorPolicyConfig(
        arm_score_min=arm_score_min,
        fire_score_floor=fire_score_floor,
        turn_down_delta=turn_down_delta,
    )


def _build_gate_config(section: dict[str, Any]) -> GateConfig:
    block_threshold = float(section.get("block_threshold"))
    if not (0.0 < block_threshold <= 1.0):
        raise ValueError("gate.block_threshold must satisfy 0 < x <= 1")
    return GateConfig(block_threshold=block_threshold)


def _build_gate_model(section: dict[str, Any]) -> GateModelConfig:
    model_section = section.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("missing required section: gate.model")
    return GateModelConfig(
        iterations=_require_positive_int(model_section.get("iterations"), "gate.model.iterations"),
        depth=_require_positive_int(model_section.get("depth"), "gate.model.depth"),
        learning_rate=_require_positive_float(model_section.get("learning_rate"), "gate.model.learning_rate"),
        l2_leaf_reg=_require_positive_float(model_section.get("l2_leaf_reg"), "gate.model.l2_leaf_reg"),
        random_seed=_require_non_negative_int(model_section.get("random_seed"), "gate.model.random_seed"),
    )
