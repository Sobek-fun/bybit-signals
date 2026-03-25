from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pump_end_v2.config import V2Config
from pump_end_v2.logging import log_info


def resolve_source_path(source_root: str | Path, path_value: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("path_value must be a non-empty string")
    source_root_path = Path(source_root).expanduser()
    raw_path = Path(path_value.strip()).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return (source_root_path / raw_path).resolve(strict=False)


def load_tabular_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"unsupported tabular extension: {suffix or '<none>'}")


def load_market_inputs(config: V2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    source_root = config.raw["data"]["source_root"]
    bars_15m_path = resolve_source_path(source_root, config.data_files.bars_15m_path)
    bars_1m_path = resolve_source_path(source_root, config.data_files.bars_1m_path)
    if not bars_15m_path.exists():
        raise FileNotFoundError(f"required input file is missing: {bars_15m_path}")
    if not bars_1m_path.exists():
        raise FileNotFoundError(f"required input file is missing: {bars_1m_path}")
    raw_15m = load_tabular_frame(bars_15m_path)
    raw_1m = load_tabular_frame(bars_1m_path)
    raw_1s: pd.DataFrame | None = None
    if config.data_files.bars_1s_path != "":
        bars_1s_path = resolve_source_path(source_root, config.data_files.bars_1s_path)
        if not bars_1s_path.exists():
            raise FileNotFoundError(f"optional input file was configured but is missing: {bars_1s_path}")
        raw_1s = load_tabular_frame(bars_1s_path)
    rows_1s = len(raw_1s) if raw_1s is not None else 0
    log_info("IO", f"input load done rows_15m={len(raw_15m)} rows_1m={len(raw_1m)} rows_1s={rows_1s}")
    return raw_15m, raw_1m, raw_1s


def save_dataframe_artifact(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return output_path
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return output_path
    raise ValueError(f"unsupported dataframe artifact extension: {suffix or '<none>'}")


def save_json_artifact(payload: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path
