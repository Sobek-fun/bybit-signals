from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any


BLOCKED_TOP_LEVEL = {
    ".git",
    ".venv",
    "__pycache__",
    "analysis_outputs",
    "catboost_info",
    "artifacts/runpod_batches",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_payload(payload: Any) -> str:
    return sha256_text(stable_json(payload))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_repo_bundle(repo_root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="runpod_baseline_", dir=str(out_dir))
    bundle = Path(tmp) / "baseline_bundle.tgz"
    with tarfile.open(bundle, mode="w:gz") as tar:
        for child in repo_root.iterdir():
            rel = child.relative_to(repo_root).as_posix()
            if rel in BLOCKED_TOP_LEVEL:
                continue
            if child.name in {".git", ".venv", "__pycache__", "analysis_outputs", "catboost_info"}:
                continue
            tar.add(child, arcname=f"{repo_root.name}/{child.name}", filter=_tar_filter)
    return bundle


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    normalized = info.name.replace("\\", "/")
    blocked_parts = ["/.git/", "/.venv/", "/__pycache__/"]
    if any(part in f"/{normalized}/" for part in blocked_parts):
        return None
    return info


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def getenv_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value
