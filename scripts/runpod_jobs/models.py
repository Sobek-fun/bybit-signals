from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStateName(str, Enum):
    PREPARING = "PREPARING"
    DEPLOYING = "DEPLOYING"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class BaselineSpec:
    baseline_id: str
    baseline_hash: str = ""
    bundle_path: str = ""
    shared_inputs_hash: str = ""
    cache_key: str = ""
    self_check_command: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineSpec":
        return cls(
            baseline_id=str(payload.get("baseline_id", "")).strip(),
            baseline_hash=str(payload.get("baseline_hash", "")).strip(),
            bundle_path=str(payload.get("bundle_path", "")).strip(),
            shared_inputs_hash=str(payload.get("shared_inputs_hash", "")).strip(),
            cache_key=str(payload.get("cache_key", "")).strip(),
            self_check_command=str(payload.get("self_check_command", "")).strip(),
        )


@dataclass(slots=True)
class DeltaSpec:
    transform_script: str = ""
    overlay_files: list[str] = field(default_factory=list)
    patch_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    command_override: str = ""
    params_override: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DeltaSpec":
        payload = payload or {}
        return cls(
            transform_script=str(payload.get("transform_script", "")).strip(),
            overlay_files=[str(x) for x in payload.get("overlay_files", [])],
            patch_files=[str(x) for x in payload.get("patch_files", [])],
            changed_files=[str(x) for x in payload.get("changed_files", [])],
            command_override=str(payload.get("command_override", "")).strip(),
            params_override=dict(payload.get("params_override", {})),
        )


@dataclass(slots=True)
class ExperimentSpec:
    exp_id: str
    pod_alias: str
    run_dir: str
    delta: DeltaSpec = field(default_factory=DeltaSpec)
    expected_checks: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            exp_id=str(payload.get("exp_id", "")).strip(),
            pod_alias=str(payload.get("pod_alias", "")).strip(),
            run_dir=str(payload.get("run_dir", "")).strip(),
            delta=DeltaSpec.from_dict(payload.get("delta")),
            expected_checks=[str(x) for x in payload.get("expected_checks", [])],
        )


@dataclass(slots=True)
class PodSpec:
    pod_id: str
    alias: str
    host: str = ""
    port: int = 22
    ssh_user: str = "root"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PodSpec":
        return cls(
            pod_id=str(payload.get("pod_id", "")).strip(),
            alias=str(payload.get("alias", "")).strip(),
            host=str(payload.get("host", "")).strip(),
            port=int(payload.get("port", 22)),
            ssh_user=str(payload.get("ssh_user", "root")),
        )


@dataclass(slots=True)
class LaunchPolicy:
    one_active_experiment_per_pod: bool = True
    tail_only: bool = False
    dry_run: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LaunchPolicy":
        payload = payload or {}
        return cls(
            one_active_experiment_per_pod=bool(payload.get("one_active_experiment_per_pod", True)),
            tail_only=bool(payload.get("tail_only", False)),
            dry_run=bool(payload.get("dry_run", False)),
        )


@dataclass(slots=True)
class ArtifactPolicy:
    include_paths: list[str] = field(default_factory=list)
    required_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ArtifactPolicy":
        payload = payload or {}
        return cls(
            include_paths=[str(x) for x in payload.get("include_paths", [])],
            required_paths=[str(x) for x in payload.get("required_paths", [])],
        )


@dataclass(slots=True)
class BatchManifest:
    batch_id: str
    created_at: str
    baseline: BaselineSpec
    shared_inputs: dict[str, Any]
    pods: list[PodSpec]
    experiments: list[ExperimentSpec]
    launch_policy: LaunchPolicy = field(default_factory=LaunchPolicy)
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchManifest":
        return cls(
            batch_id=str(payload.get("batch_id", "")).strip(),
            created_at=str(payload.get("created_at", now_utc_iso())),
            baseline=BaselineSpec.from_dict(dict(payload.get("baseline", {}))),
            shared_inputs=dict(payload.get("shared_inputs", {})),
            pods=[PodSpec.from_dict(x) for x in payload.get("pods", [])],
            experiments=[ExperimentSpec.from_dict(x) for x in payload.get("experiments", [])],
            launch_policy=LaunchPolicy.from_dict(payload.get("launch_policy")),
            artifact_policy=ArtifactPolicy.from_dict(payload.get("artifact_policy")),
        )


@dataclass(slots=True)
class PodAssignment:
    exp_id: str
    pod_id: str
    pod_alias: str


@dataclass(slots=True)
class LaunchResult:
    exp_id: str
    pod_id: str
    pod_alias: str
    host: str
    port: int
    run_dir: str
    release_dir: str
    log_path: str
    tail_command: str
    state_path: str
    launched_at: str = field(default_factory=now_utc_iso)


@dataclass(slots=True)
class RunState:
    exp_id: str
    state: RunStateName
    started_at: str = ""
    finished_at: str = ""
    last_heartbeat: str = ""
    pid: int = 0
    exit_code: int | None = None
    run_dir: str = ""
    log_path: str = ""
    message: str = ""


@dataclass(slots=True)
class ArtifactManifest:
    exp_id: str
    files: list[str]
    ready: bool
    generated_at: str = field(default_factory=now_utc_iso)


@dataclass(slots=True)
class CacheManifest:
    baseline_hash: str
    cache_key: str
    files: list[str]
    created_at: str = field(default_factory=now_utc_iso)
