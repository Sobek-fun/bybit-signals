import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return output_path
