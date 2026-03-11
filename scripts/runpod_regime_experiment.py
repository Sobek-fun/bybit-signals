from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


class RunpodError(RuntimeError):
    pass


class SSHError(RuntimeError):
    pass


@dataclass
class PodConnection:
    pod_id: str
    host: str
    port: int
    ssh_user: str
    ssh_key_path: Path


class RunpodClient:
    def __init__(self, api_key: str, base_url: str = "https://rest.runpod.io/v1"):
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise RunpodError("RUNPOD_API_KEY is not set")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: int = 45,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                headers=self._headers(),
                json=payload,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RunpodError(f"RunPod request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RunpodError(f"RunPod {method} {path} failed: {response.status_code} {response.text[:600]}")
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except Exception:
            return response.text

    def create_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_id: str,
        cloud_type: str,
        interruptible: bool,
        container_disk_gb: int,
        volume_gb: int,
        min_vcpu_per_gpu: int,
        min_ram_per_gpu: int,
        ports: list[str],
        network_volume_id: str | None,
        volume_mount_path: str | None,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "computeType": "GPU",
            "cloudType": cloud_type,
            "gpuCount": 1,
            "gpuTypeIds": [gpu_type_id],
            "imageName": image_name,
            "containerDiskInGb": int(container_disk_gb),
            "volumeInGb": int(volume_gb),
            "minVCPUPerGPU": int(min_vcpu_per_gpu),
            "minRAMPerGPU": int(min_ram_per_gpu),
            "ports": ports,
            "interruptible": bool(interruptible),
            "supportPublicIp": True,
            "env": {},
        }
        if network_volume_id:
            payload["networkVolumeId"] = network_volume_id
        if volume_mount_path:
            payload["volumeMountPath"] = volume_mount_path
        data = self._request("POST", "/pods", payload)
        if not isinstance(data, dict) or not data.get("id"):
            raise RunpodError(f"Unexpected create pod response: {data}")
        return data

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        data = self._request("GET", f"/pods/{pod_id}")
        if not isinstance(data, dict):
            raise RunpodError(f"Unexpected pod response: {data}")
        return data

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def wait_for_ssh(self, pod_id: str, timeout_seconds: int = 1200, poll_seconds: int = 8) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_state: dict[str, Any] = {}
        while time.time() < deadline:
            state = self.get_pod(pod_id)
            last_state = state
            desired = str(state.get("desiredStatus") or "")
            ip = state.get("publicIp")
            port_map = state.get("portMappings") or {}
            ssh_port = None
            if isinstance(port_map, dict):
                ssh_port = port_map.get("22")
                if ssh_port is None:
                    ssh_port = port_map.get(22)
            if desired == "RUNNING" and ip and ssh_port:
                return {"id": pod_id, "host": str(ip), "port": int(ssh_port), "raw": state}
            time.sleep(poll_seconds)
        raise RunpodError(f"Timed out waiting for pod SSH readiness: {pod_id}, last_state={last_state}")


def _run_cmd(cmd: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    if check and proc.returncode != 0:
        out = (proc.stderr or proc.stdout or "").strip()
        raise SSHError(out or f"command failed: {' '.join(cmd)}")
    return proc


def _base_ssh_cmd(conn: PodConnection) -> list[str]:
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


def _base_scp_cmd(conn: PodConnection) -> list[str]:
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


def run_ssh(conn: PodConnection, remote_command: str, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run_cmd(_base_ssh_cmd(conn) + [remote_command], timeout=timeout, check=check)


def scp_to_remote(conn: PodConnection, local_path: Path, remote_path: str) -> None:
    _run_cmd(_base_scp_cmd(conn) + [str(local_path), f"{conn.ssh_user}@{conn.host}:{remote_path}"], check=True)


def scp_from_remote(conn: PodConnection, remote_path: str, local_path: Path, recursive: bool = True) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _base_scp_cmd(conn)
    if recursive:
        cmd.append("-r")
    cmd += [f"{conn.ssh_user}@{conn.host}:{remote_path}", str(local_path)]
    _run_cmd(cmd, check=True)


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = tarinfo.name.replace("\\", "/")
    blocked = ["/.git/", "/.venv/", "/__pycache__/", "/analysis_outputs/", "/catboost_info/", "/signals/"]
    for item in blocked:
        if item in f"/{name}/":
            return None
    return tarinfo


def make_repo_bundle(repo_root: Path) -> Path:
    tmpdir = tempfile.mkdtemp(prefix="runpod_regime_bundle_")
    bundle = Path(tmpdir) / "bybit_signals_repo.tgz"
    with tarfile.open(bundle, mode="w:gz") as tar:
        tar.add(repo_root, arcname=repo_root.name, filter=_tar_filter)
    return bundle


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime experiment on RunPod")
    parser.add_argument("--runpod-api-key", type=str, default=os.getenv("RUNPOD_API_KEY", "").strip(), required=False)
    parser.add_argument("--runpod-api-base", type=str, default="https://rest.runpod.io/v1")
    parser.add_argument("--storage-id", type=str, default="e4sm7sqxod")
    parser.add_argument("--storage-name", type=str, default="cautious_indigo_reindeer")
    parser.add_argument("--storage-mount-path", type=str, default="/workspace")
    parser.add_argument("--disable-network-volume", action="store_true")
    parser.add_argument("--image-name", type=str, default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--volume-gb", type=int, default=0)
    parser.add_argument("--min-vcpu-per-gpu", type=int, default=4)
    parser.add_argument("--min-ram-per-gpu", type=int, default=16)
    parser.add_argument("--ssh-user", type=str, default="root")
    parser.add_argument("--ssh-key-path", type=str, default=str(Path("~/.ssh/id_ed25519").expanduser()))
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--remote-project-base", type=str, default="/workspace")
    parser.add_argument("--runtime-minutes", type=int, default=5)
    parser.add_argument("--create-retries", type=int, default=6)
    parser.add_argument("--create-retry-sleep-seconds", type=int, default=20)
    parser.add_argument("--terminate-pod", action="store_true", default=True)
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "8")))
    parser.add_argument("--train-end", type=str, default="2026-02-20")
    parser.add_argument("--oos-start", type=str, default="2026-02-21 00:00:00")
    parser.add_argument("--oos-end", type=str, default="2026-03-03 23:59:59")
    parser.add_argument("--start-date", type=str, default="2025-01-01 00:00:00")
    parser.add_argument("--detector-dir", type=str, default="artifacts/tune_threshold_no_argmax_liq7d_detector")
    parser.add_argument("--tokens-file", type=str, default="config/regime_tokens_curated55.txt")
    parser.add_argument("--run-name", type=str, default=f"exp_curated55_from_2025_01_01_{now_utc()}")
    parser.add_argument("--clickhouse-user", type=str, default="admin")
    parser.add_argument("--clickhouse-pass", type=str, default="GtChrHFvAL3CybQB")
    parser.add_argument("--clickhouse-host", type=str, default="185.189.45.79")
    parser.add_argument("--clickhouse-port", type=int, default=8123)
    parser.add_argument("--clickhouse-db", type=str, default="bybit")
    return parser.parse_args()


def choose_pod_variant() -> list[dict[str, Any]]:
    return [
        {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY", "interruptible": True, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY", "interruptible": False, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY", "interruptible": True, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY", "interruptible": False, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "SECURE", "interruptible": True, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "SECURE", "interruptible": True, "ram": 16, "vcpu": 4},
        {"gpu": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA RTX 4000 Ada Generation", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA RTX A4000", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA RTX A4500", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA RTX A5000", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA RTX A6000", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA L4", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA A40", "cloud": "COMMUNITY", "interruptible": True, "ram": 12, "vcpu": 2},
        {"gpu": "NVIDIA A5000", "cloud": "SECURE", "interruptible": True, "ram": 12, "vcpu": 2},
    ]


def build_pipeline_command(remote_repo_dir: str, args: argparse.Namespace) -> str:
    ch_dsn = f"http://{args.clickhouse_user}:{args.clickhouse_pass}@{args.clickhouse_host}:{args.clickhouse_port}/{args.clickhouse_db}"
    run_root = f"{args.storage_mount_path.rstrip('/')}/experiments/{args.run_name}"
    detector_dir = f"{remote_repo_dir}/{args.detector_dir}".replace("\\", "/")
    tokens_file = f"{remote_repo_dir}/{args.tokens_file}".replace("\\", "/")
    export_symbols_flag = f"--symbols-file \\\"$TOKENS_FILE\\\" " if args.tokens_file else ""
    build_symbols_flag = f"--symbols-file \\\"$TOKENS_FILE\\\" " if args.tokens_file else ""
    fixed_universe_flag = f"--fixed-universe-file \\\"$TOKENS_FILE\\\" " if args.tokens_file else ""
    timeout_seconds = max(60, int(args.runtime_minutes) * 60)
    quoted_ch = shlex.quote(ch_dsn)
    quoted_run = shlex.quote(run_root)
    quoted_detector = shlex.quote(detector_dir)
    quoted_tokens = shlex.quote(tokens_file)
    quoted_workers = shlex.quote(str(args.workers))
    quoted_train_end = shlex.quote(args.train_end)
    quoted_oos_start = shlex.quote(args.oos_start)
    quoted_oos_end = shlex.quote(args.oos_end)
    quoted_start_date = shlex.quote(args.start_date)
    cmd = (
        f"set -euo pipefail; "
        f"export CH_DB={quoted_ch}; "
        f"export RUN_ROOT={quoted_run}; "
        f"export DETECTOR_DIR={quoted_detector}; "
        f"export TOKENS_FILE={quoted_tokens}; "
        f"export WORKERS={quoted_workers}; "
        f"export TRAIN_END={quoted_train_end}; "
        f"export OOS_START={quoted_oos_start}; "
        f"export OOS_END={quoted_oos_end}; "
        f"mkdir -p \"$RUN_ROOT\"; "
        f"cd {shlex.quote(remote_repo_dir)}; "
        f"if ! command -v uv >/dev/null 2>&1; then "
        f"python -m pip install --upgrade pip >/tmp/regime_bootstrap.log 2>&1; "
        f"python -m pip install uv >>/tmp/regime_bootstrap.log 2>&1; "
        f"fi; "
        f"timeout {timeout_seconds}s bash -lc "
        f"\"uv run --python 3.13 python -m pump_end_threshold.cli.export_pump_end_signals "
        f"--start-date {quoted_start_date} "
        f"--end-date \\\"$OOS_END\\\" "
        f"--clickhouse-dsn \\\"$CH_DB\\\" "
        f"--model-dir \\\"$DETECTOR_DIR\\\" "
        f"{export_symbols_flag}"
        f"--run-dir \\\"$RUN_ROOT\\\" "
        f"--output \\\"$RUN_ROOT/final_signals.csv\\\" "
        f"--raw-signals-output \\\"$RUN_ROOT/raw_detector_signals_curated55.parquet\\\" "
        f"--skip-guard "
        f"--workers \\\"$WORKERS\\\" && "
        f"uv run --python 3.13 python -m pump_end_threshold.cli.build_regime_dataset "
        f"--clickhouse-dsn \\\"$CH_DB\\\" "
        f"--signals-path \\\"$RUN_ROOT/raw_detector_signals_curated55.parquet\\\" "
        f"{build_symbols_flag}"
        f"{fixed_universe_flag}"
        f"--run-dir \\\"$RUN_ROOT\\\" "
        f"--top-n-universe 55 "
        f"--tp-pct 4.5 "
        f"--sl-pct 10.0 "
        f"--max-horizon-bars 200 "
        f"--trade-replay-source 1s "
        f"--target-col target_pause_value_next_12h "
        f"--target-profile pause_value_12h_v2_curated "
        f"--target-min-resolved 3 "
        f"--target-sl-rate-threshold 0.55 "
        f"--feature-profile regime_compact_v4 && "
        f"uv run --python 3.13 python -m pump_end_threshold.cli.train_regime_guard "
        f"--run-dir \\\"$RUN_ROOT\\\" "
        f"--dataset-parquet \\\"$RUN_ROOT/regime_dataset.parquet\\\" "
        f"--target-col target_pause_value_next_12h "
        f"--train-end \\\"$TRAIN_END\\\" "
        f"--time-budget-min 45 "
        f"--fold-days 21 "
        f"--min-train-days 120 "
        f"--embargo-hours 12 "
        f"--iterations 3500 "
        f"--early-stopping-rounds 250 "
        f"--seed 42 "
        f"--score-mode comprehensive "
        f"--max-blocked-share 0.15 "
        f"--min-signal-keep-rate 0.80 "
        f"--min-valid-folds 6 "
        f"--policy-grid low "
        f"--disable-auto-class-weights\""
    )
    return cmd


def run() -> int:
    args = parse_args()
    if not args.runpod_api_key:
        raise SystemExit("RUNPOD API key is required")
    repo_root = Path(args.project_root).expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"project root not found: {repo_root}")
    if args.tokens_file:
        local_tokens = (repo_root / args.tokens_file).resolve()
        if not local_tokens.exists():
            print(json.dumps({"event": "tokens_file_missing", "path": str(local_tokens)}, ensure_ascii=False))
            args.tokens_file = ""
    ssh_key_path = Path(args.ssh_key_path).expanduser().resolve()
    if not ssh_key_path.exists():
        raise SystemExit(f"ssh key not found: {ssh_key_path}")

    client = RunpodClient(api_key=args.runpod_api_key, base_url=args.runpod_api_base)
    pod_id = ""
    conn: PodConnection | None = None
    bundle: Path | None = None
    should_delete = True
    run_name = args.run_name
    remote_repo_dir = f"{args.remote_project_base.rstrip('/')}/{repo_root.name}"
    run_root_remote = f"{args.storage_mount_path.rstrip('/')}/experiments/{run_name}"
    local_artifacts_dir = repo_root / "artifacts" / "runpod_exports" / run_name
    local_artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        created = None
        errors: list[str] = []
        pod_name = f"regime-{run_name}"[:120]
        for attempt in range(max(1, int(args.create_retries))):
            for variant in choose_pod_variant():
                try:
                    created = client.create_pod(
                        name=pod_name,
                        image_name=args.image_name,
                        gpu_type_id=variant["gpu"],
                        cloud_type=variant["cloud"],
                        interruptible=variant["interruptible"],
                        container_disk_gb=args.container_disk_gb,
                        volume_gb=args.volume_gb,
                        min_vcpu_per_gpu=int(variant.get("vcpu", args.min_vcpu_per_gpu)),
                        min_ram_per_gpu=int(variant.get("ram", args.min_ram_per_gpu)),
                        ports=["22/tcp", "8888/http"],
                        network_volume_id=None if args.disable_network_volume else args.storage_id,
                        volume_mount_path=None if args.disable_network_volume else args.storage_mount_path,
                    )
                    created["request_variant"] = {**variant, "attempt": attempt + 1}
                    break
                except Exception as exc:
                    errors.append(f"attempt={attempt + 1} variant={variant}: {exc}")
                    continue
            if created:
                break
            if attempt + 1 < max(1, int(args.create_retries)):
                time.sleep(max(1, int(args.create_retry_sleep_seconds)))
        if not created:
            raise RunpodError("All create pod variants failed: " + " | ".join(errors))

        pod_id = str(created["id"])
        print(json.dumps({"event": "pod_created", "pod_id": pod_id, "variant": created.get("request_variant")}, ensure_ascii=False))
        ready = client.wait_for_ssh(pod_id)
        conn = PodConnection(
            pod_id=pod_id,
            host=str(ready["host"]),
            port=int(ready["port"]),
            ssh_user=args.ssh_user,
            ssh_key_path=ssh_key_path,
        )
        print(json.dumps({"event": "pod_ssh_ready", "host": conn.host, "port": conn.port}, ensure_ascii=False))

        run_ssh(conn, f"mkdir -p {shlex.quote(args.remote_project_base)} {shlex.quote(args.storage_mount_path.rstrip('/') + '/experiments')}", check=True)
        bundle = make_repo_bundle(repo_root)
        remote_bundle = f"/tmp/{bundle.name}"
        scp_to_remote(conn, bundle, remote_bundle)
        run_ssh(
            conn,
            " && ".join(
                [
                    f"rm -rf {shlex.quote(remote_repo_dir)}",
                    f"mkdir -p {shlex.quote(args.remote_project_base)}",
                    f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(args.remote_project_base)}",
                    f"rm -f {shlex.quote(remote_bundle)}",
                    f"mkdir -p {shlex.quote(run_root_remote)}",
                ]
            ),
            check=True,
        )
        print(json.dumps({"event": "repo_uploaded", "remote_repo_dir": remote_repo_dir}, ensure_ascii=False))

        remote_log = f"{run_root_remote}/pipeline.log"
        pipeline_cmd = build_pipeline_command(remote_repo_dir, args)
        wrapped = f"bash -lc {shlex.quote(pipeline_cmd)} > {shlex.quote(remote_log)} 2>&1"
        proc = run_ssh(conn, wrapped, timeout=max(600, args.runtime_minutes * 60 + 300), check=False)
        print(json.dumps({"event": "pipeline_finished", "returncode": proc.returncode, "remote_log": remote_log}, ensure_ascii=False))

        tail = run_ssh(conn, f"if [ -f {shlex.quote(remote_log)} ]; then tail -n 200 {shlex.quote(remote_log)}; fi", check=False)
        (local_artifacts_dir / "pipeline_tail.log").write_text((tail.stdout or "") + "\n" + (tail.stderr or ""), encoding="utf-8")

        summary = {
            "pod_id": pod_id,
            "run_name": run_name,
            "remote_run_dir": run_root_remote,
            "remote_log": remote_log,
            "returncode": proc.returncode,
            "storage_id": args.storage_id,
            "storage_name": args.storage_name,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (local_artifacts_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "summary_saved", "path": str(local_artifacts_dir / 'summary.json')}, ensure_ascii=False))

        try:
            scp_from_remote(conn, remote_log, local_artifacts_dir / "pipeline.log", recursive=False)
        except Exception:
            pass

        if proc.returncode != 0:
            if args.keep_pod_on_failure:
                should_delete = False
            raise SystemExit(f"Pipeline exited with code {proc.returncode}. See {local_artifacts_dir / 'pipeline_tail.log'}")

        return 0
    finally:
        try:
            if bundle and bundle.exists():
                bundle.unlink(missing_ok=True)
                tmp = bundle.parent
                if tmp.exists():
                    try:
                        tmp.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass
        if pod_id and args.terminate_pod and should_delete:
            try:
                client.delete_pod(pod_id)
                print(json.dumps({"event": "pod_terminated", "pod_id": pod_id}, ensure_ascii=False))
            except Exception as exc:
                print(json.dumps({"event": "pod_terminate_failed", "pod_id": pod_id, "error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130)
