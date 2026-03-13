from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LocalBatchPaths:
    root: Path
    manifest: Path
    launch_results: Path
    status: Path
    downloaded: Path
    logs: Path


def local_batch_paths(repo_root: Path, batch_id: str) -> LocalBatchPaths:
    root = repo_root / "artifacts" / "runpod_batches" / batch_id
    return LocalBatchPaths(
        root=root,
        manifest=root / "batch_manifest.json",
        launch_results=root / "launch_results.json",
        status=root / "batch_status.json",
        downloaded=root / "downloaded",
        logs=root / "logs",
    )


@dataclass(slots=True)
class RemoteExperimentPaths:
    run_dir: str
    release_dir: str
    venv_dir: str
    tmp_dir: str
    log_path: str
    state_path: str
    artifacts_manifest_path: str


def remote_paths(batch_id: str, exp_id: str, run_dir: str, tmp_root: str = "/tmp/bybit-signals") -> RemoteExperimentPaths:
    exp_tmp = f"{tmp_root.rstrip('/')}/{batch_id}/{exp_id}"
    release_dir = f"{exp_tmp}/release"
    venv_dir = f"{exp_tmp}/venv"
    return RemoteExperimentPaths(
        run_dir=run_dir,
        release_dir=release_dir,
        venv_dir=venv_dir,
        tmp_dir=f"{exp_tmp}/tmp",
        log_path=f"{run_dir.rstrip('/')}/pipeline.log",
        state_path=f"{run_dir.rstrip('/')}/run_state.json",
        artifacts_manifest_path=f"{run_dir.rstrip('/')}/artifacts_manifest.json",
    )
