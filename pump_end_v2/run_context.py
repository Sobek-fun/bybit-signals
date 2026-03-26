from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    resolved_run_id = run_id or created_at.strftime("run_%Y%m%d_%H%M%S")
    manager = ArtifactManager(runs_root)
    run_dir = manager.create_run_dir(resolved_run_id)
    return RunContext(
        run_id=resolved_run_id,
        run_dir=run_dir,
        config_path=Path(config_path),
        created_at=created_at,
    )
