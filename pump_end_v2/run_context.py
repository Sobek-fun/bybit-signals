from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from pump_end_v2.artifacts import ArtifactManager


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    config_path: Path
    created_at: datetime


def create_run_context(
    config_path: str | Path, runs_root: str | Path, run_id: str | None = None
) -> RunContext:
    created_at = datetime.now(UTC)
    config_run_id = _build_config_run_id(config_path)
    base_run_id = run_id or config_run_id
    resolved_run_id = _ensure_unique_run_id(runs_root, base_run_id)
    manager = ArtifactManager(runs_root)
    run_dir = manager.create_run_dir(resolved_run_id)
    return RunContext(
        run_id=resolved_run_id,
        run_dir=run_dir,
        config_path=Path(config_path),
        created_at=created_at,
    )


def _build_config_run_id(config_path: str | Path) -> str:
    stem = Path(config_path).stem.strip()
    if not stem:
        return "run"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return normalized or "run"


def _ensure_unique_run_id(runs_root: str | Path, base_run_id: str) -> str:
    root = Path(runs_root)
    candidate = base_run_id
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base_run_id}__{suffix}"
        suffix += 1
    return candidate
