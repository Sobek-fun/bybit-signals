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
    tp_row_weight: float
    sl_row_weight: float
    timeout_row_weight: float


@dataclass(frozen=True, slots=True)
class DetectorModelConfig:
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_seed: int
    hidden_channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    fit_eval_fraction: float
    fit_eval_min_rows: int
    sequence_learning_rate: float
    weight_decay: float
    pre_episode_context_bars: int
    decision_window_bars: int


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
class DetectorPolicySearchConfig:
    arm_candidates: tuple[float, ...]
    fire_candidates: tuple[float, ...]
    turn_candidates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GateConfig:
    block_threshold: float


@dataclass(frozen=True, slots=True)
class GateThresholdSearchConfig:
    threshold_candidates: tuple[float, ...]
    include_disabled_candidate: bool
    target_sl_capture_model: float
    max_tp_tax_model: float
    min_blocked_trainable: int
    require_sl_gt_tp: bool
    min_blocked_share_model: float | None
    max_blocked_share_model: float | None
    min_signals_per_30d_after_execution: float | None
    max_signals_per_30d_after_execution: float | None


@dataclass(frozen=True, slots=True)
class GateModelConfig:
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_seed: int
    tp_row_weight: float
    sl_row_weight: float


@dataclass(frozen=True, slots=True)
class GateModelSearchConfig:
    depth_candidates: tuple[int, ...]
    l2_leaf_reg_candidates: tuple[float, ...]
    random_seed_candidates: tuple[int, ...]
    tp_row_weight_candidates: tuple[float, ...]
    learning_rate_iteration_pairs: tuple[tuple[float, int], ...]


@dataclass(frozen=True, slots=True)
class ReferenceSymbolsConfig:
    btc_symbol: str
    eth_symbol: str


@dataclass(frozen=True, slots=True)
class DataWindowConfig:
    start: datetime
    end: datetime | None


@dataclass(frozen=True, slots=True)
class DataUniverseConfig:
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DataClickHouseConfig:
    candles_table: str
    transactions_table: str
    timezone: str


@dataclass(frozen=True, slots=True)
class V2Config:
    raw: dict[str, Any]
    splits: SplitBounds
    references: ReferenceSymbolsConfig
    data_window: DataWindowConfig
    data_universe: DataUniverseConfig
    data_clickhouse: DataClickHouseConfig
    event_opener: EventOpenerConfig
    resolver: ResolverConfig
    detector_model: DetectorModelConfig
    detector_cv: DetectorCVConfig
    detector_policy: DetectorPolicyConfig
    search_detector_policy: DetectorPolicySearchConfig | None
    gate_config: GateConfig
    search_gate_threshold: GateThresholdSearchConfig | None
    search_gate_model: GateModelSearchConfig | None
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
    _validate_execution(config["execution"])
    _validate_gate(config["gate"], config["execution"])
    _validate_compute(config["compute"])
    splits = _build_splits(config["splits"])
    references = _build_references(config["data"]["references"])
    data_window = _build_data_window(config["data"]["window"])
    data_universe = _build_data_universe(config["data"]["universe"])
    data_clickhouse = _build_data_clickhouse(config["data"]["clickhouse"])
    event_opener = _build_event_opener(config["data"]["event_opener"])
    resolver = _build_resolver(config["data"]["resolver"])
    detector_model = _build_detector_model(config["detector"])
    detector_cv = _build_detector_cv(config["detector"])
    detector_policy = _build_detector_policy(config["detector"])
    search_detector_policy = _build_detector_policy_search(config.get("search"))
    execution = ExecutionContract(
        tp_pct=float(config["execution"]["tp_pct"]),
        sl_pct=float(config["execution"]["sl_pct"]),
        max_hold_bars=int(config["execution"]["max_hold_bars"]),
        entry_shift_bars=int(config["execution"]["entry_shift_bars"]),
    )
    gate_config = _build_gate_config(config["gate"])
    search_gate_threshold = _build_gate_threshold_search(config.get("search"))
    search_gate_model = _build_gate_model_search(config.get("search"))
    gate_model = _build_gate_model(config["gate"], execution)
    return V2Config(
        raw=copy.deepcopy(config),
        splits=splits,
        references=references,
        data_window=data_window,
        data_universe=data_universe,
        data_clickhouse=data_clickhouse,
        event_opener=event_opener,
        resolver=resolver,
        detector_model=detector_model,
        detector_cv=detector_cv,
        detector_policy=detector_policy,
        search_detector_policy=search_detector_policy,
        gate_config=gate_config,
        search_gate_threshold=search_gate_threshold,
        search_gate_model=search_gate_model,
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


def _parse_date_start(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be YYYY-MM-DD string")
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
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
    event_opener_section = data_section.get("event_opener")
    if not isinstance(event_opener_section, dict):
        raise ValueError("missing required section: data.event_opener")
    resolver_section = data_section.get("resolver")
    if not isinstance(resolver_section, dict):
        raise ValueError("missing required section: data.resolver")
    references_section = data_section.get("references")
    if not isinstance(references_section, dict):
        raise ValueError("missing required section: data.references")
    window_section = data_section.get("window")
    if not isinstance(window_section, dict):
        raise ValueError("missing required section: data.window")
    universe_section = data_section.get("universe")
    if not isinstance(universe_section, dict):
        raise ValueError("missing required section: data.universe")
    clickhouse_section = data_section.get("clickhouse")
    if not isinstance(clickhouse_section, dict):
        raise ValueError("missing required section: data.clickhouse")
    _validate_event_opener(event_opener_section)
    _validate_resolver(resolver_section)
    _validate_references(references_section)
    _validate_data_window(window_section)
    _validate_data_universe(universe_section)
    _validate_data_clickhouse(clickhouse_section)


def _validate_detector(detector_section: dict[str, Any]) -> None:
    _validate_non_negative(detector_section, "detector")
    _build_detector_model(detector_section)
    _build_detector_cv(detector_section)
    _build_detector_policy(detector_section)


def _validate_gate(gate_section: dict[str, Any], execution_section: dict[str, Any]) -> None:
    _build_gate_config(gate_section)
    execution = ExecutionContract(
        tp_pct=float(execution_section["tp_pct"]),
        sl_pct=float(execution_section["sl_pct"]),
        max_hold_bars=int(execution_section["max_hold_bars"]),
        entry_shift_bars=int(execution_section["entry_shift_bars"]),
    )
    _build_gate_model(gate_section, execution)


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
        min_runup_pct=_require_positive_float(
            section.get("min_runup_pct"), "data.event_opener.min_runup_pct"
        ),
        near_high_lookback_bars=_require_positive_int(
            section.get("near_high_lookback_bars"),
            "data.event_opener.near_high_lookback_bars",
        ),
        near_high_tol_pct=_require_non_negative_float(
            section.get("near_high_tol_pct"), "data.event_opener.near_high_tol_pct"
        ),
        volume_ratio_lookback_bars=_require_positive_int(
            section.get("volume_ratio_lookback_bars"),
            "data.event_opener.volume_ratio_lookback_bars",
        ),
        min_volume_ratio=_require_positive_float(
            section.get("min_volume_ratio"), "data.event_opener.min_volume_ratio"
        ),
        max_episode_bars=_require_positive_int(
            section.get("max_episode_bars"), "data.event_opener.max_episode_bars"
        ),
        expiry_drawdown_pct=_require_positive_float(
            section.get("expiry_drawdown_pct"), "data.event_opener.expiry_drawdown_pct"
        ),
        cooldown_bars=_require_non_negative_int(
            section.get("cooldown_bars"), "data.event_opener.cooldown_bars"
        ),
    )


def _build_resolver(section: dict[str, Any]) -> ResolverConfig:
    horizon_bars = _require_positive_int(
        section.get("horizon_bars"), "data.resolver.horizon_bars"
    )
    max_wait_bars_for_success = _require_positive_int(
        section.get("max_wait_bars_for_success"),
        "data.resolver.max_wait_bars_for_success",
    )
    if max_wait_bars_for_success > horizon_bars:
        raise ValueError(
            "data.resolver.max_wait_bars_for_success must be <= data.resolver.horizon_bars"
        )
    return ResolverConfig(
        horizon_bars=horizon_bars,
        success_pullback_pct=_require_positive_float(
            section.get("success_pullback_pct"), "data.resolver.success_pullback_pct"
        ),
        max_prepullback_squeeze_pct=_require_positive_float(
            section.get("max_prepullback_squeeze_pct"),
            "data.resolver.max_prepullback_squeeze_pct",
        ),
        flat_max_abs_move_pct=_require_positive_float(
            section.get("flat_max_abs_move_pct"), "data.resolver.flat_max_abs_move_pct"
        ),
        max_wait_bars_for_success=max_wait_bars_for_success,
        tp_row_weight=_require_positive_float(
            section.get("tp_row_weight", 1.0), "data.resolver.tp_row_weight"
        ),
        sl_row_weight=_require_positive_float(
            section.get("sl_row_weight", 1.5), "data.resolver.sl_row_weight"
        ),
        timeout_row_weight=_require_positive_float(
            section.get("timeout_row_weight", 0.5), "data.resolver.timeout_row_weight"
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
    if btc_symbol.strip() == eth_symbol.strip():
        raise ValueError(
            "data.references.btc_symbol must differ from data.references.eth_symbol"
        )
    return ReferenceSymbolsConfig(
        btc_symbol=btc_symbol.strip().upper(),
        eth_symbol=eth_symbol.strip().upper(),
    )


def _validate_data_window(section: dict[str, Any]) -> None:
    _build_data_window(section)


def _build_data_window(section: dict[str, Any]) -> DataWindowConfig:
    start = _parse_date_start(section.get("start"), "data.window.start")
    end_raw = section.get("end")
    end = None
    if end_raw is not None:
        end = _parse_date(end_raw, "data.window.end")
        if end <= start:
            raise ValueError("data.window.end must be greater than data.window.start")
    return DataWindowConfig(start=start, end=end)


def _validate_data_universe(section: dict[str, Any]) -> None:
    _build_data_universe(section)


def _build_data_universe(section: dict[str, Any]) -> DataUniverseConfig:
    raw_symbols = section.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ValueError("data.universe.symbols must be a non-empty list")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError("data.universe.symbols must contain non-empty strings")
        symbol = raw_symbol.strip().upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    if not normalized:
        raise ValueError("data.universe.symbols must contain at least one symbol")
    return DataUniverseConfig(symbols=tuple(normalized))


def _validate_data_clickhouse(section: dict[str, Any]) -> None:
    _build_data_clickhouse(section)


def _build_data_clickhouse(section: dict[str, Any]) -> DataClickHouseConfig:
    candles_table = section.get("candles_table")
    if not isinstance(candles_table, str) or not candles_table.strip():
        raise ValueError("data.clickhouse.candles_table must be a non-empty string")
    transactions_table = section.get("transactions_table")
    if not isinstance(transactions_table, str) or not transactions_table.strip():
        raise ValueError("data.clickhouse.transactions_table must be a non-empty string")
    timezone = section.get("timezone")
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("data.clickhouse.timezone must be a non-empty string")
    return DataClickHouseConfig(
        candles_table=candles_table.strip(),
        transactions_table=transactions_table.strip(),
        timezone=timezone.strip(),
    )


def _build_detector_model(section: dict[str, Any]) -> DetectorModelConfig:
    model_section = section.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("missing required section: detector.model")
    return DetectorModelConfig(
        iterations=_require_positive_int(
            model_section.get("iterations"), "detector.model.iterations"
        ),
        depth=_require_positive_int(model_section.get("depth"), "detector.model.depth"),
        learning_rate=_require_positive_float(
            model_section.get("learning_rate"), "detector.model.learning_rate"
        ),
        l2_leaf_reg=_require_positive_float(
            model_section.get("l2_leaf_reg"), "detector.model.l2_leaf_reg"
        ),
        random_seed=_require_non_negative_int(
            model_section.get("random_seed"), "detector.model.random_seed"
        ),
        hidden_channels=_require_positive_int(
            model_section.get("hidden_channels", 64),
            "detector.model.hidden_channels",
        ),
        kernel_size=_require_positive_int(
            model_section.get("kernel_size", 3),
            "detector.model.kernel_size",
        ),
        dilations=_parse_positive_int_tuple(
            model_section.get("dilations", [1, 2, 4]),
            "detector.model.dilations",
        ),
        dropout=_require_bounded_float(
            model_section.get("dropout", 0.10),
            "detector.model.dropout",
            lower=0.0,
            upper=1.0,
            inclusive_lower=True,
            inclusive_upper=True,
        ),
        batch_size=_require_positive_int(
            model_section.get("batch_size", 256),
            "detector.model.batch_size",
        ),
        max_epochs=_require_positive_int(
            model_section.get("max_epochs", 40),
            "detector.model.max_epochs",
        ),
        early_stopping_patience=_require_positive_int(
            model_section.get("early_stopping_patience", 5),
            "detector.model.early_stopping_patience",
        ),
        fit_eval_fraction=_require_bounded_float(
            model_section.get("fit_eval_fraction", 0.2),
            "detector.model.fit_eval_fraction",
            lower=0.0,
            upper=0.5,
            inclusive_lower=False,
            inclusive_upper=False,
        ),
        fit_eval_min_rows=_require_positive_int(
            model_section.get("fit_eval_min_rows", 128),
            "detector.model.fit_eval_min_rows",
        ),
        sequence_learning_rate=_require_positive_float(
            model_section.get("sequence_learning_rate", 1e-3),
            "detector.model.sequence_learning_rate",
        ),
        weight_decay=_require_non_negative_float(
            model_section.get("weight_decay", 1e-4),
            "detector.model.weight_decay",
        ),
        pre_episode_context_bars=_require_positive_int(
            model_section.get("pre_episode_context_bars", 16),
            "detector.model.pre_episode_context_bars",
        ),
        decision_window_bars=_require_positive_int(
            model_section.get("decision_window_bars", 6),
            "detector.model.decision_window_bars",
        ),
    )


def _parse_positive_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    out: list[int] = []
    for item in value:
        out.append(_require_positive_int(item, field_name))
    return tuple(out)


def _require_bounded_float(
    value: Any,
    field_name: str,
    *,
    lower: float,
    upper: float,
    inclusive_lower: bool,
    inclusive_upper: bool,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    lower_ok = parsed >= lower if inclusive_lower else parsed > lower
    upper_ok = parsed <= upper if inclusive_upper else parsed < upper
    if not (lower_ok and upper_ok):
        lower_mark = "[" if inclusive_lower else "("
        upper_mark = "]" if inclusive_upper else ")"
        raise ValueError(f"{field_name} must be in range {lower_mark}{lower}, {upper}{upper_mark}")
    return parsed


def _build_detector_cv(section: dict[str, Any]) -> DetectorCVConfig:
    cv_section = section.get("cv")
    if not isinstance(cv_section, dict):
        raise ValueError("missing required section: detector.cv")
    return DetectorCVConfig(
        min_train_days=_require_positive_int(
            cv_section.get("min_train_days"), "detector.cv.min_train_days"
        ),
        fold_span_days=_require_positive_int(
            cv_section.get("fold_span_days"), "detector.cv.fold_span_days"
        ),
        max_folds=_require_positive_int(
            cv_section.get("max_folds"), "detector.cv.max_folds"
        ),
    )


def _build_detector_policy(section: dict[str, Any]) -> DetectorPolicyConfig:
    arm_score_min = float(section.get("arm_score_min"))
    fire_score_floor = float(section.get("fire_score_floor"))
    turn_down_delta = float(section.get("turn_down_delta"))
    if not (0.0 < arm_score_min <= 1.0):
        raise ValueError("detector.arm_score_min must satisfy 0 < x <= 1")
    if not (0.0 <= fire_score_floor <= arm_score_min):
        raise ValueError(
            "detector.fire_score_floor must satisfy 0 <= x <= detector.arm_score_min"
        )
    if not (0.0 < turn_down_delta <= 1.0):
        raise ValueError("detector.turn_down_delta must satisfy 0 < x <= 1")
    return DetectorPolicyConfig(
        arm_score_min=arm_score_min,
        fire_score_floor=fire_score_floor,
        turn_down_delta=turn_down_delta,
    )


def _parse_float_candidates(
    value: Any, field_name: str, *, allow_empty: bool = True
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain numeric values") from exc
    if not allow_empty and not out:
        raise ValueError(f"{field_name} must be non-empty")
    return tuple(out)


def _build_detector_policy_search(
    search_section: Any,
) -> DetectorPolicySearchConfig | None:
    if search_section is None:
        return None
    if not isinstance(search_section, dict):
        raise ValueError("search section must be a table")
    detector_policy_section = search_section.get("detector_policy")
    if detector_policy_section is None:
        return None
    if not isinstance(detector_policy_section, dict):
        raise ValueError("search.detector_policy must be a table")
    arm_candidates = _parse_float_candidates(
        detector_policy_section.get("arm_candidates", []),
        "search.detector_policy.arm_candidates",
    )
    fire_candidates = _parse_float_candidates(
        detector_policy_section.get("fire_candidates", []),
        "search.detector_policy.fire_candidates",
    )
    turn_candidates = _parse_float_candidates(
        detector_policy_section.get("turn_candidates", []),
        "search.detector_policy.turn_candidates",
    )
    return DetectorPolicySearchConfig(
        arm_candidates=arm_candidates,
        fire_candidates=fire_candidates,
        turn_candidates=turn_candidates,
    )


def _build_gate_config(section: dict[str, Any]) -> GateConfig:
    block_threshold = float(section.get("block_threshold"))
    if not (0.0 < block_threshold <= 1.0):
        raise ValueError("gate.block_threshold must satisfy 0 < x <= 1")
    return GateConfig(block_threshold=block_threshold)


def _build_gate_threshold_search(
    search_section: Any,
) -> GateThresholdSearchConfig | None:
    if search_section is None:
        return None
    if not isinstance(search_section, dict):
        raise ValueError("search section must be a table")
    gate_threshold_section = search_section.get("gate_threshold")
    if gate_threshold_section is None:
        return None
    if not isinstance(gate_threshold_section, dict):
        raise ValueError("search.gate_threshold must be a table")
    threshold_candidates = _parse_float_candidates(
        gate_threshold_section.get("threshold_candidates", []),
        "search.gate_threshold.threshold_candidates",
    )
    include_disabled_candidate = bool(
        gate_threshold_section.get("include_disabled_candidate", False)
    )
    target_sl_capture_model = float(
        gate_threshold_section.get("target_sl_capture_model", 0.20)
    )
    if not (0.0 <= target_sl_capture_model <= 1.0):
        raise ValueError(
            "search.gate_threshold.target_sl_capture_model must satisfy 0 <= x <= 1"
        )
    max_tp_tax_model = float(gate_threshold_section.get("max_tp_tax_model", 0.10))
    if not (0.0 <= max_tp_tax_model <= 1.0):
        raise ValueError(
            "search.gate_threshold.max_tp_tax_model must satisfy 0 <= x <= 1"
        )
    min_blocked_trainable = _require_non_negative_int(
        gate_threshold_section.get("min_blocked_trainable", 10),
        "search.gate_threshold.min_blocked_trainable",
    )
    require_sl_gt_tp = bool(gate_threshold_section.get("require_sl_gt_tp", True))
    min_blocked_share_model = _parse_optional_bounded_float(
        gate_threshold_section.get("min_blocked_share_model"),
        "search.gate_threshold.min_blocked_share_model",
        lower=0.0,
        upper=1.0,
    )
    max_blocked_share_model = _parse_optional_bounded_float(
        gate_threshold_section.get("max_blocked_share_model"),
        "search.gate_threshold.max_blocked_share_model",
        lower=0.0,
        upper=1.0,
    )
    if (
        min_blocked_share_model is not None
        and max_blocked_share_model is not None
        and min_blocked_share_model > max_blocked_share_model
    ):
        raise ValueError(
            "search.gate_threshold.min_blocked_share_model must be <= search.gate_threshold.max_blocked_share_model"
        )
    min_signals_per_30d_after_execution = _parse_optional_bounded_float(
        gate_threshold_section.get("min_signals_per_30d_after_execution"),
        "search.gate_threshold.min_signals_per_30d_after_execution",
        lower=0.0,
        upper=None,
    )
    max_signals_per_30d_after_execution = _parse_optional_bounded_float(
        gate_threshold_section.get("max_signals_per_30d_after_execution"),
        "search.gate_threshold.max_signals_per_30d_after_execution",
        lower=0.0,
        upper=None,
    )
    if (
        min_signals_per_30d_after_execution is not None
        and max_signals_per_30d_after_execution is not None
        and min_signals_per_30d_after_execution > max_signals_per_30d_after_execution
    ):
        raise ValueError(
            "search.gate_threshold.min_signals_per_30d_after_execution must be <= search.gate_threshold.max_signals_per_30d_after_execution"
        )
    return GateThresholdSearchConfig(
        threshold_candidates=threshold_candidates,
        include_disabled_candidate=include_disabled_candidate,
        target_sl_capture_model=target_sl_capture_model,
        max_tp_tax_model=max_tp_tax_model,
        min_blocked_trainable=min_blocked_trainable,
        require_sl_gt_tp=require_sl_gt_tp,
        min_blocked_share_model=min_blocked_share_model,
        max_blocked_share_model=max_blocked_share_model,
        min_signals_per_30d_after_execution=min_signals_per_30d_after_execution,
        max_signals_per_30d_after_execution=max_signals_per_30d_after_execution,
    )


def _parse_optional_bounded_float(
    value: Any,
    field_name: str,
    *,
    lower: float | None,
    upper: float | None,
) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or null") from exc
    if lower is not None and parsed < float(lower):
        raise ValueError(f"{field_name} must be >= {lower}")
    if upper is not None and parsed > float(upper):
        raise ValueError(f"{field_name} must be <= {upper}")
    return parsed


def _build_gate_model(
    section: dict[str, Any], execution_contract: ExecutionContract
) -> GateModelConfig:
    model_section = section.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("missing required section: gate.model")
    sl_row_weight = _require_positive_float(
        model_section.get("sl_row_weight", 1.0), "gate.model.sl_row_weight"
    )
    default_tp_row_weight = (
        float(execution_contract.tp_pct) / float(execution_contract.sl_pct)
        if float(execution_contract.sl_pct) > 0.0
        else 1.0
    )
    tp_row_weight = _require_positive_float(
        model_section.get("tp_row_weight", default_tp_row_weight),
        "gate.model.tp_row_weight",
    )
    return GateModelConfig(
        iterations=_require_positive_int(
            model_section.get("iterations"), "gate.model.iterations"
        ),
        depth=_require_positive_int(model_section.get("depth"), "gate.model.depth"),
        learning_rate=_require_positive_float(
            model_section.get("learning_rate"), "gate.model.learning_rate"
        ),
        l2_leaf_reg=_require_positive_float(
            model_section.get("l2_leaf_reg"), "gate.model.l2_leaf_reg"
        ),
        random_seed=_require_non_negative_int(
            model_section.get("random_seed"), "gate.model.random_seed"
        ),
        tp_row_weight=tp_row_weight,
        sl_row_weight=sl_row_weight,
    )


def _parse_positive_int_candidates(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: list[int] = []
    for item in value:
        parsed = _require_positive_int(item, field_name)
        out.append(parsed)
    return tuple(out)


def _parse_non_negative_int_candidates(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: list[int] = []
    for item in value:
        parsed = _require_non_negative_int(item, field_name)
        out.append(parsed)
    return tuple(out)


def _build_gate_model_search(search_section: Any) -> GateModelSearchConfig | None:
    if search_section is None:
        return None
    if not isinstance(search_section, dict):
        raise ValueError("search section must be a table")
    gate_model_section = search_section.get("gate_model")
    if gate_model_section is None:
        return None
    if not isinstance(gate_model_section, dict):
        raise ValueError("search.gate_model must be a table")
    depth_candidates = _parse_positive_int_candidates(
        gate_model_section.get("depth_candidates", []),
        "search.gate_model.depth_candidates",
    )
    l2_leaf_reg_candidates = _parse_float_candidates(
        gate_model_section.get("l2_leaf_reg_candidates", []),
        "search.gate_model.l2_leaf_reg_candidates",
    )
    for value in l2_leaf_reg_candidates:
        if float(value) <= 0.0:
            raise ValueError(
                "search.gate_model.l2_leaf_reg_candidates must contain positive values"
            )
    random_seed_candidates = _parse_non_negative_int_candidates(
        gate_model_section.get("random_seed_candidates", []),
        "search.gate_model.random_seed_candidates",
    )
    tp_row_weight_candidates = _parse_float_candidates(
        gate_model_section.get("tp_row_weight_candidates", []),
        "search.gate_model.tp_row_weight_candidates",
    )
    for value in tp_row_weight_candidates:
        if float(value) <= 0.0:
            raise ValueError(
                "search.gate_model.tp_row_weight_candidates must contain positive values"
            )
    raw_pairs = gate_model_section.get("learning_rate_iteration_pairs", [])
    if not isinstance(raw_pairs, list):
        raise ValueError(
            "search.gate_model.learning_rate_iteration_pairs must be a list"
        )
    learning_rate_iteration_pairs: list[tuple[float, int]] = []
    for idx, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            raise ValueError(
                "search.gate_model.learning_rate_iteration_pairs entries must be tables"
            )
        learning_rate = _require_positive_float(
            pair.get("learning_rate"),
            f"search.gate_model.learning_rate_iteration_pairs[{idx}].learning_rate",
        )
        iterations = _require_positive_int(
            pair.get("iterations"),
            f"search.gate_model.learning_rate_iteration_pairs[{idx}].iterations",
        )
        learning_rate_iteration_pairs.append((learning_rate, iterations))
    return GateModelSearchConfig(
        depth_candidates=depth_candidates,
        l2_leaf_reg_candidates=l2_leaf_reg_candidates,
        random_seed_candidates=random_seed_candidates,
        tp_row_weight_candidates=tp_row_weight_candidates,
        learning_rate_iteration_pairs=tuple(learning_rate_iteration_pairs),
    )
