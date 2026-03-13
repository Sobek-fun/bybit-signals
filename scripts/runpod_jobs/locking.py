from __future__ import annotations

import time
from pathlib import Path


def acquire_lock(lock_path: Path, timeout_seconds: int = 30, poll_seconds: float = 0.2) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            lock_path.mkdir(parents=True, exist_ok=False)
            return True
        except FileExistsError:
            time.sleep(poll_seconds)
    return False


def release_lock(lock_path: Path) -> None:
    if lock_path.exists() and lock_path.is_dir():
        lock_path.rmdir()


def cache_complete(cache_dir: Path, required: list[str]) -> bool:
    marker = cache_dir / "done.marker"
    manifest = cache_dir / "cache_manifest.json"
    if not marker.exists() or not manifest.exists():
        return False
    return all((cache_dir / rel).exists() for rel in required)
