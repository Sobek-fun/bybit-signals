from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class RunpodError(RuntimeError):
    pass


class SSHError(RuntimeError):
    pass


@dataclass
class PodConn:
    pod_id: str
    host: str
    port: int
    ssh_user: str
    ssh_key_path: Path


@dataclass
class ExperimentSpec:
    exp_id: str
    run_root: str
    transform_script: str = ""
    target_col: str = ""


GPU_VARIANTS: tuple[dict[str, Any], ...] = (
    {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY", "interruptible": True, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY", "interruptible": False, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "SECURE", "interruptible": True, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY", "interruptible": True, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY", "interruptible": False, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "SECURE", "interruptible": True, "vcpu": 4, "ram": 16},
    {"gpu": "NVIDIA RTX A4500", "cloud": "COMMUNITY", "interruptible": True, "vcpu": 2, "ram": 12},
    {"gpu": "NVIDIA RTX 4000 Ada Generation", "cloud": "COMMUNITY", "interruptible": True, "vcpu": 2, "ram": 12},
    {"gpu": "NVIDIA L4", "cloud": "COMMUNITY", "interruptible": True, "vcpu": 2, "ram": 12},
)


class RunpodClient:
    def __init__(self, api_key: str, base_url: str = "https://rest.runpod.io/v1"):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise RunpodError("RUNPOD_API_KEY is required")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            resp = requests.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=45,
            )
        except requests.RequestException as exc:
            raise RunpodError(f"RunPod request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise RunpodError(f"RunPod {method} {path} failed: {resp.status_code} {resp.text[:800]}")
        if resp.status_code == 204:
            return None
        try:
            return resp.json()
        except Exception:
            return resp.text

    def create_pod(
        self,
        *,
        name: str,
        gpu_type_id: str,
        image_name: str,
        cloud_type: str,
        interruptible: bool,
        container_disk_gb: int,
        min_vcpu_per_gpu: int,
        min_ram_per_gpu: int,
        network_volume_id: str,
        volume_mount_path: str,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "computeType": "GPU",
            "cloudType": cloud_type,
            "gpuCount": 1,
            "gpuTypeIds": [gpu_type_id],
            "imageName": image_name,
            "containerDiskInGb": int(container_disk_gb),
            "volumeInGb": 0,
            "minVCPUPerGPU": int(min_vcpu_per_gpu),
            "minRAMPerGPU": int(min_ram_per_gpu),
            "ports": ["22/tcp", "8888/http"],
            "interruptible": bool(interruptible),
            "supportPublicIp": True,
            "env": {},
            "networkVolumeId": network_volume_id,
            "volumeMountPath": volume_mount_path,
        }
        data = self._request("POST", "/pods", payload)
        if not isinstance(data, dict) or not data.get("id"):
            raise RunpodError(f"Unexpected pod create response: {data}")
        return data

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        data = self._request("GET", f"/pods/{pod_id}")
        if not isinstance(data, dict):
            raise RunpodError(f"Unexpected pod response: {data}")
        return data

    def wait_for_ssh(self, pod_id: str, timeout_seconds: int = 1500, poll_seconds: int = 8) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_state: dict[str, Any] = {}
        while time.time() < deadline:
            state = self.get_pod(pod_id)
            last_state = state
            desired = str(state.get("desiredStatus") or "")
            ip = state.get("publicIp")
            port_map = state.get("portMappings") or {}
            ssh_port = port_map.get("22") if isinstance(port_map, dict) else None
            if ssh_port is None and isinstance(port_map, dict):
                ssh_port = port_map.get(22)
            if desired == "RUNNING" and ip and ssh_port:
                return {"host": str(ip), "port": int(ssh_port), "raw": state}
            time.sleep(poll_seconds)
        raise RunpodError(f"Timed out waiting for SSH: {pod_id}, last={last_state}")


def _run_local(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise SSHError((proc.stderr or proc.stdout or "").strip())
    return proc


def _ssh_base(conn: PodConn) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(conn.port),
        "-i",
        str(conn.ssh_key_path),
        f"{conn.ssh_user}@{conn.host}",
    ]


def _scp_base(conn: PodConn) -> list[str]:
    return [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(conn.port),
        "-i",
        str(conn.ssh_key_path),
    ]


def run_ssh(conn: PodConn, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run_local(_ssh_base(conn) + [command], check=check)


def scp_to_remote(conn: PodConn, local_path: Path, remote_path: str) -> None:
    _run_local(_scp_base(conn) + [str(local_path), f"{conn.ssh_user}@{conn.host}:{remote_path}"])


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = tarinfo.name.replace("\\", "/")
    for blocked in ("/.git/", "/.venv/", "/__pycache__/", "/analysis_outputs/", "/catboost_info/"):
        if blocked in f"/{name}/":
            return None
    return tarinfo


def make_bundle(repo_root: Path) -> Path:
    tmp = tempfile.mkdtemp(prefix="runpod_regime_batch_")
    bundle = Path(tmp) / "repo.tgz"
    with tarfile.open(bundle, mode="w:gz") as tar:
        tar.add(repo_root, arcname=repo_root.name, filter=_tar_filter)
    return bundle


def start_remote_nohup(conn: PodConn, command: str, log_path: str) -> int:
    wrapped = f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
    proc = run_ssh(conn, wrapped, check=True)
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        raise SSHError("failed to read remote pid")
    return int(lines[-1].strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic RunPod launcher for regime experiments")
    parser.add_argument("--runpod-api-key", required=True)
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--storage-id", default="e4sm7sqxod")
    parser.add_argument("--storage-mount-path", default="/workspace")
    parser.add_argument("--image-name", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-key-path", default=str(Path("~/.ssh/id_ed25519").expanduser()))
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--remote-project-base", default="/workspace")
    parser.add_argument("--container-disk-gb", type=int, default=60)
    parser.add_argument("--create-retries", type=int, default=12)
    parser.add_argument("--create-retry-sleep-seconds", type=int, default=15)
    parser.add_argument("--detector-dir", default="artifacts/tune_threshold_no_argmax_liq7d_detector")
    parser.add_argument("--tokens-file", default="config/regime_tokens_curated55.txt")
    parser.add_argument("--clickhouse-dsn", default="http://admin:GtChrHFvAL3CybQB@185.189.45.79:8123/bybit")
    return parser.parse_args()


def load_specs(path: Path) -> list[ExperimentSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("spec file must contain a JSON array")
    out: list[ExperimentSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        exp_id = str(item.get("exp_id", "")).strip()
        run_root = str(item.get("run_root", "")).strip()
        if not exp_id or not run_root:
            raise SystemExit("each spec item must contain exp_id and run_root")
        out.append(
            ExperimentSpec(
                exp_id=exp_id,
                run_root=run_root,
                transform_script=str(item.get("transform_script", "")).strip(),
                target_col=str(item.get("target_col", "")).strip(),
            )
        )
    if not out:
        raise SystemExit("spec file has no experiments")
    return out


def create_pod_with_retries(client: RunpodClient, args: argparse.Namespace, pod_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    for attempt in range(1, args.create_retries + 1):
        for variant in GPU_VARIANTS:
            try:
                created = client.create_pod(
                    name=pod_name[:120],
                    gpu_type_id=str(variant["gpu"]),
                    image_name=args.image_name,
                    cloud_type=str(variant["cloud"]),
                    interruptible=bool(variant["interruptible"]),
                    container_disk_gb=args.container_disk_gb,
                    min_vcpu_per_gpu=int(variant["vcpu"]),
                    min_ram_per_gpu=int(variant["ram"]),
                    network_volume_id=args.storage_id,
                    volume_mount_path=args.storage_mount_path,
                )
                return created, {**variant, "attempt": attempt}
            except Exception as exc:
                errors.append(f"attempt={attempt} variant={variant}: {exc}")
        time.sleep(args.create_retry_sleep_seconds)
    raise RunpodError("Cannot create pod: " + " | ".join(errors))


def build_command(
    *,
    spec: ExperimentSpec,
    remote_repo: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    run_root = spec.run_root
    log_path = f"{run_root.rstrip('/')}/pipeline.log"
    transform_arg = shlex.quote(spec.transform_script) if spec.transform_script else ""
    target_arg = shlex.quote(spec.target_col) if spec.target_col else ""
    command = " ".join(
        [
            shlex.quote(f"{remote_repo}/scripts/runpod_jobs/run_experiment.sh"),
            shlex.quote(spec.exp_id),
            shlex.quote(run_root),
            shlex.quote(remote_repo),
            shlex.quote(args.clickhouse_dsn),
            shlex.quote(f"{remote_repo}/{args.detector_dir}".replace("\\", "/")),
            shlex.quote(f"{remote_repo}/{args.tokens_file}".replace("\\", "/")),
            transform_arg,
            target_arg,
        ]
    )
    return command, log_path


def launch_one(
    *,
    client: RunpodClient,
    args: argparse.Namespace,
    repo_root: Path,
    bundle: Path,
    ssh_key_path: Path,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    pod_name = f"regime-{spec.exp_id}-{int(time.time())}"
    created, variant = create_pod_with_retries(client, args, pod_name)
    pod_id = str(created["id"])
    ready = client.wait_for_ssh(pod_id)
    conn = PodConn(
        pod_id=pod_id,
        host=str(ready["host"]),
        port=int(ready["port"]),
        ssh_user=args.ssh_user,
        ssh_key_path=ssh_key_path,
    )
    remote_repo = f"{args.remote_project_base.rstrip('/')}/{repo_root.name}"
    remote_bundle = f"/tmp/{bundle.name}.{spec.exp_id}.tgz"
    scp_to_remote(conn, bundle, remote_bundle)
    run_ssh(
        conn,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(args.remote_project_base)}",
                f"rm -rf {shlex.quote(remote_repo)}",
                f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(args.remote_project_base)}",
                f"rm -f {shlex.quote(remote_bundle)}",
                f"mkdir -p {shlex.quote(spec.run_root)}",
                f"sed -i 's/\\r$//' {shlex.quote(remote_repo + '/scripts/runpod_jobs/run_experiment.sh')}",
                f"chmod +x {shlex.quote(remote_repo + '/scripts/runpod_jobs/run_experiment.sh')}",
            ]
        ),
    )
    command, log_path = build_command(spec=spec, remote_repo=remote_repo, args=args)
    pid = start_remote_nohup(conn, command, log_path)
    return {
        "exp_id": spec.exp_id,
        "pod_id": pod_id,
        "host": conn.host,
        "port": conn.port,
        "remote_pid": pid,
        "run_root": spec.run_root,
        "log_path": log_path,
        "variant": variant,
        "tail_command": (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {conn.port} "
            f"-i \"{conn.ssh_key_path}\" {conn.ssh_user}@{conn.host} "
            f"\"tail -n 200 -f {shlex.quote(log_path)}\""
        ),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.project_root).expanduser().resolve()
    ssh_key_path = Path(args.ssh_key_path).expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"project root not found: {repo_root}")
    if not ssh_key_path.exists():
        raise SystemExit(f"ssh key not found: {ssh_key_path}")
    spec_file = Path(args.spec_file).expanduser().resolve()
    if not spec_file.exists():
        raise SystemExit(f"spec file not found: {spec_file}")

    specs = load_specs(spec_file)
    client = RunpodClient(api_key=args.runpod_api_key)
    bundle = make_bundle(repo_root)
    results: list[dict[str, Any]] = []
    try:
        for spec in specs:
            result = launch_one(
                client=client,
                args=args,
                repo_root=repo_root,
                bundle=bundle,
                ssh_key_path=ssh_key_path,
                spec=spec,
            )
            results.append(result)
            print(json.dumps({"event": "experiment_started", **result}, ensure_ascii=False), flush=True)
    finally:
        if bundle.exists():
            parent = bundle.parent
            bundle.unlink(missing_ok=True)
            try:
                parent.rmdir()
            except Exception:
                pass

    out = repo_root / "artifacts" / "runpod_exports" / f"launched_batch_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"spec_file": str(spec_file), "experiments": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "launch_complete", "summary_path": str(out), "count": len(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
