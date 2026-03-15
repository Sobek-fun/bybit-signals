from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BatchRuntime:
    workspace_root: str
    requirements_file: str
    pipeline_command: str
    clickhouse_dsn_env: str
    detector_dir_remote: str
    tokens_file_remote: str
    python_bin: str = "python3"
    setup_command: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchRuntime":
        return cls(
            workspace_root=str(payload.get("workspace_root", "/workspace/experiments")).strip(),
            requirements_file=str(payload.get("requirements_file", "scripts/runpod_jobs/requirements_runpod.txt")).strip(),
            pipeline_command=str(payload.get("pipeline_command", "")).strip(),
            clickhouse_dsn_env=str(payload.get("clickhouse_dsn_env", "CH_DB")).strip(),
            detector_dir_remote=str(payload.get("detector_dir_remote", "")).strip(),
            tokens_file_remote=str(payload.get("tokens_file_remote", "")).strip(),
            python_bin=str(payload.get("python_bin", "python3")).strip(),
            setup_command=str(payload.get("setup_command", "")).strip(),
            extra_env={str(k): str(v) for k, v in dict(payload.get("extra_env", {})).items()},
        )


@dataclass(slots=True)
class ExperimentSpec:
    exp_id: str
    pod_alias: str
    patch_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            exp_id=str(payload.get("exp_id", "")).strip(),
            pod_alias=str(payload.get("pod_alias", "")).strip(),
            patch_files=[str(x) for x in payload.get("patch_files", [])],
        )


@dataclass(slots=True)
class PodSpec:
    alias: str
    pod_id: str = ""
    host: str = ""
    port: int = 22
    ssh_user: str = "root"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PodSpec":
        return cls(
            alias=str(payload.get("alias", "")).strip(),
            pod_id=str(payload.get("pod_id", "")).strip(),
            host=str(payload.get("host", "")).strip(),
            port=int(payload.get("port", 22)),
            ssh_user=str(payload.get("ssh_user", "root")).strip() or "root",
        )


@dataclass(slots=True)
class BatchSpec:
    batch_id: str
    runtime: BatchRuntime
    experiments: list[ExperimentSpec]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchSpec":
        return cls(
            batch_id=str(payload.get("batch_id", "")).strip(),
            runtime=BatchRuntime.from_dict(dict(payload.get("runtime", {}))),
            experiments=[ExperimentSpec.from_dict(x) for x in payload.get("experiments", [])],
        )


@dataclass(slots=True)
class LaunchResult:
    exp_id: str
    pod_alias: str
    pod_id: str
    host: str
    port: int
    exp_root: str
    tail_command: str
    launch_command: str
