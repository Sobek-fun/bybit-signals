from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANONICAL_DIRS = (
    "prepared",
    "detector",
    "gate",
    "eval/val",
    "eval/test",
    "reports",
)


class ArtifactManager:
    def __init__(self, runs_root: str | Path):
        self.runs_root = Path(runs_root)

    def create_run_dir(self, run_id: str | None = None) -> Path:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        resolved_run_id = run_id or datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
        run_dir = self.runs_root / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for relative_dir in CANONICAL_DIRS:
            (run_dir / relative_dir).mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_config_snapshot(self, run_dir: str | Path, config_path: str | Path) -> Path:
        target = Path(run_dir) / "config.snapshot.toml"
        source = Path(config_path)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    def save_run_manifest(self, run_dir: str | Path, manifest: dict[str, Any]) -> Path:
        target = Path(run_dir) / "run_manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
        return target

    def stage_output_dir(self, run_dir: str | Path, stage_name: str, split: str | None = None) -> Path:
        root = Path(run_dir)
        if stage_name == "prepared":
            return root / "prepared"
        if stage_name == "detector":
            return root / "detector"
        if stage_name == "gate":
            return root / "gate"
        if stage_name == "reports":
            return root / "reports"
        if stage_name == "eval":
            if split not in {"val", "test"}:
                raise ValueError("eval split must be one of: val, test")
            return root / "eval" / split
        raise ValueError(f"unknown stage_name: {stage_name}")
