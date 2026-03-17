from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SHARED_VENV_DIR = "/workspace/.venvs/bybit-signals-runpod"


@dataclass(slots=True)
class LocalPaths:
    root: Path
    launch_results: Path
    assembled_root: Path


def local_paths(repo_root: Path, batch_id: str) -> LocalPaths:
    root = repo_root / "artifacts" / "runpod_launch" / batch_id
    return LocalPaths(
        root=root,
        launch_results=root / "launch_results.json",
        assembled_root=root / "assembled",
    )


@dataclass(slots=True)
class RemoteExperimentPaths:
    exp_root: str
    src_dir: str
    venv_dir: str
    run_dir: str
    log_path: str
    started_at_path: str
    finished_at_path: str
    exit_code_path: str
    launch_command_path: str
    start_script_path: str


def remote_paths(workspace_root: str, batch_id: str, exp_id: str) -> RemoteExperimentPaths:
    batch_root = f"{workspace_root.rstrip('/')}/{batch_id}"
    exp_root = f"{batch_root}/{exp_id}"
    return RemoteExperimentPaths(
        exp_root=exp_root,
        src_dir=f"{exp_root}/src",
        venv_dir=SHARED_VENV_DIR,
        run_dir=f"{exp_root}/run",
        log_path=f"{exp_root}/pipeline.log",
        started_at_path=f"{exp_root}/started_at.txt",
        finished_at_path=f"{exp_root}/finished_at.txt",
        exit_code_path=f"{exp_root}/exit_code.txt",
        launch_command_path=f"{exp_root}/launch_command.txt",
        start_script_path=f"{exp_root}/start.sh",
    )
