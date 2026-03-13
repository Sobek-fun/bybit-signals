from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.runpod_jobs.models import (
    ArtifactManifest,
    BaselineSpec,
    BatchManifest,
    CacheManifest,
    DeltaSpec,
    ExperimentSpec,
    LaunchResult,
    LaunchPolicy,
    PodSpec,
    RunState,
    RunStateName,
    now_utc_iso,
)
from scripts.runpod_jobs.paths import local_batch_paths, remote_paths
from scripts.runpod_jobs.runpod_api import RunpodClient
from scripts.runpod_jobs.ssh import PodConn, check_ssh_ready, run_ssh, scp_from_remote, scp_to_remote
from scripts.runpod_jobs.utils import compute_file_sha256, ensure_dir, hash_payload, make_repo_bundle, read_json, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> BatchManifest:
    payload = read_json(path)
    manifest = BatchManifest.from_dict(payload)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: BatchManifest) -> None:
    if not manifest.batch_id:
        raise SystemExit("batch_id is required")
    if not manifest.baseline.baseline_id:
        raise SystemExit("baseline.baseline_id is required")
    if not manifest.experiments:
        raise SystemExit("experiments is empty")
    exp_ids = [x.exp_id for x in manifest.experiments]
    if len(exp_ids) != len(set(exp_ids)):
        raise SystemExit("exp_id must be unique")
    aliases = [x.alias for x in manifest.pods]
    if len(aliases) != len(set(aliases)):
        raise SystemExit("pod aliases must be unique")
    pod_alias_set = set(aliases)
    for exp in manifest.experiments:
        if not exp.pod_alias:
            raise SystemExit(f"{exp.exp_id}: pod_alias is required")
        if exp.pod_alias not in pod_alias_set:
            raise SystemExit(f"{exp.exp_id}: unknown pod_alias={exp.pod_alias}")


def _manifest_to_dict(manifest: BatchManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["experiments"] = [asdict(x) for x in manifest.experiments]
    payload["pods"] = [asdict(x) for x in manifest.pods]
    return payload


def prepare_batch(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec_file).expanduser().resolve()
    if not spec_path.exists():
        raise SystemExit(f"spec file not found: {spec_path}")
    raw = read_json(spec_path)
    batch_id = str(raw.get("batch_id") or args.batch_id or _default_batch_id())

    local = local_batch_paths(REPO_ROOT, batch_id)
    ensure_dir(local.root)
    ensure_dir(local.logs)
    ensure_dir(local.downloaded)
    ensure_dir(local.root / "baseline")
    ensure_dir(local.root / "delta_manifests")

    bundle = make_repo_bundle(REPO_ROOT, local.root / "baseline")
    bundle_hash = compute_file_sha256(bundle)
    shared_inputs = dict(raw.get("shared_inputs", {}))
    shared_inputs_hash = hash_payload(shared_inputs)
    baseline_hash = hash_payload({"bundle_sha256": bundle_hash, "shared_inputs_hash": shared_inputs_hash})
    baseline = BaselineSpec.from_dict(
        {
            **dict(raw.get("baseline", {})),
            "baseline_hash": baseline_hash,
            "bundle_path": str(bundle),
            "shared_inputs_hash": shared_inputs_hash,
            "cache_key": f"{baseline_hash[:24]}",
        }
    )
    baseline_file = local.root / "baseline" / "baseline_manifest.json"
    write_json(baseline_file, asdict(baseline))

    pods = [PodSpec.from_dict(x) for x in raw.get("pods", [])]
    experiments: list[ExperimentSpec] = []
    for item in raw.get("experiments", []):
        exp = ExperimentSpec.from_dict(item)
        if not exp.run_dir:
            exp.run_dir = f"/workspace/experiments/{batch_id}/{exp.exp_id}"
        experiments.append(exp)
        write_json(local.root / "delta_manifests" / f"{exp.exp_id}.json", asdict(exp.delta))

    manifest = BatchManifest(
        batch_id=batch_id,
        created_at=now_utc_iso(),
        baseline=baseline,
        shared_inputs=shared_inputs,
        pods=pods,
        experiments=experiments,
        launch_policy=LaunchPolicy.from_dict(raw.get("launch_policy")),
        artifact_policy=BatchManifest.from_dict(raw).artifact_policy,
    )
    validate_manifest(manifest)
    write_json(local.manifest, _manifest_to_dict(manifest))
    write_json(
        local.root / "dry_run_plan.json",
        {
            "batch_id": batch_id,
            "baseline_hash": baseline_hash,
            "experiments": [
                {
                    "exp_id": exp.exp_id,
                    "pod_alias": exp.pod_alias,
                    "run_dir": exp.run_dir,
                    "delta": asdict(exp.delta),
                }
                for exp in experiments
            ],
        },
    )
    print(str(local.manifest))
    return 0


def _default_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _load_inventory(path: Path) -> dict[str, PodSpec]:
    payload = read_json(path)
    items = payload.get("pods", []) if isinstance(payload, dict) else payload
    out: dict[str, PodSpec] = {}
    for item in items:
        pod = PodSpec.from_dict(item)
        if not pod.alias:
            continue
        out[pod.alias] = pod
    return out


def _resolve_conn(api: RunpodClient, pod: PodSpec, ssh_key: Path) -> PodConn:
    endpoint = api.wait_for_ssh_endpoint(pod.pod_id, timeout_seconds=600, poll_seconds=6)
    conn = PodConn(
        pod_id=pod.pod_id,
        host=endpoint.host,
        port=endpoint.port,
        ssh_user=pod.ssh_user or "root",
        ssh_key_path=ssh_key,
    )
    if not check_ssh_ready(conn):
        raise SystemExit(f"SSH endpoint is not ready for pod={pod.pod_id}")
    return conn


def _build_experiment_command(manifest: BatchManifest, exp: ExperimentSpec, rp: Any) -> str:
    si = manifest.shared_inputs
    transform_script = exp.delta.transform_script or "scripts/runpod_jobs/transform_template.py"
    command_override = exp.delta.command_override
    params_override = json.dumps(exp.delta.params_override, ensure_ascii=False)
    return " ".join(
        [
            shlex.quote(f"{rp.release_dir}/scripts/runpod_jobs/run_experiment.sh"),
            "--exp-id",
            shlex.quote(exp.exp_id),
            "--run-dir",
            shlex.quote(rp.run_dir),
            "--release-dir",
            shlex.quote(rp.release_dir),
            "--venv-dir",
            shlex.quote(rp.venv_dir),
            "--tmp-dir",
            shlex.quote(rp.tmp_dir),
            "--baseline-hash",
            shlex.quote(manifest.baseline.baseline_hash),
            "--cache-root",
            shlex.quote("/workspace/experiments/_baseline_cache"),
            "--locks-root",
            shlex.quote("/workspace/experiments/_locks"),
            "--clickhouse-dsn",
            shlex.quote(str(si.get("clickhouse_dsn", os.getenv("CLICKHOUSE_DSN", "")))),
            "--detector-dir",
            shlex.quote(str(si.get("detector_dir", "artifacts/tune_threshold_no_argmax_liq7d_detector"))),
            "--tokens-file",
            shlex.quote(str(si.get("tokens_file", "config/regime_tokens_curated55.txt"))),
            "--transform-script",
            shlex.quote(transform_script),
            "--target-col",
            shlex.quote(str(exp.delta.params_override.get("target_col", ""))),
            "--train-end",
            shlex.quote(str(si.get("train_end", ""))),
            "--oos-end",
            shlex.quote(str(si.get("oos_end", ""))),
            "--start-date",
            shlex.quote(str(si.get("start_date", ""))),
            "--build-dataset-args-json",
            shlex.quote(json.dumps(si.get("build_dataset_args", {}), ensure_ascii=False)),
            "--train-args-json",
            shlex.quote(json.dumps(si.get("train_args_default", {}), ensure_ascii=False)),
            "--params-override-json",
            shlex.quote(params_override),
            "--command-override",
            shlex.quote(command_override),
        ]
    )


def _deploy_release(conn: PodConn, manifest: BatchManifest, exp: ExperimentSpec, rp: Any) -> None:
    remote_bundle = f"{rp.tmp_dir}/baseline_bundle.tgz"
    run_ssh(
        conn,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(rp.tmp_dir)} {shlex.quote(rp.run_dir)}",
                f"rm -rf {shlex.quote(rp.release_dir)}",
                f"mkdir -p {shlex.quote(rp.release_dir)}",
            ]
        ),
    )
    scp_to_remote(conn, Path(manifest.baseline.bundle_path), remote_bundle)
    run_ssh(
        conn,
        " && ".join(
            [
                f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(rp.release_dir)} --strip-components=1",
                f"rm -f {shlex.quote(remote_bundle)}",
                f"chmod +x {shlex.quote(rp.release_dir + '/scripts/runpod_jobs/run_experiment.sh')}",
            ]
        ),
    )
    for rel in exp.delta.overlay_files + exp.delta.changed_files:
        local = REPO_ROOT / rel
        if not local.exists():
            raise SystemExit(f"Overlay file not found: {local}")
        remote = f"{rp.release_dir}/{rel.replace('\\', '/')}"
        run_ssh(conn, f"mkdir -p {shlex.quote(str(Path(remote).parent).replace('\\\\', '/'))}")
        scp_to_remote(conn, local, remote)
    for patch_file in exp.delta.patch_files:
        local_patch = REPO_ROOT / patch_file
        if not local_patch.exists():
            raise SystemExit(f"Patch file not found: {local_patch}")
        remote_patch = f"{rp.tmp_dir}/{Path(patch_file).name}"
        scp_to_remote(conn, local_patch, remote_patch)
        run_ssh(
            conn,
            " && ".join(
                [
                    f"cd {shlex.quote(rp.release_dir)}",
                    f"patch -p1 < {shlex.quote(remote_patch)}",
                ]
            ),
        )


def _write_remote_launch_manifest(conn: PodConn, exp: ExperimentSpec, rp: Any, manifest: BatchManifest) -> None:
    payload = {
        "batch_id": manifest.batch_id,
        "exp_id": exp.exp_id,
        "run_dir": rp.run_dir,
        "release_dir": rp.release_dir,
        "venv_dir": rp.venv_dir,
        "delta": asdict(exp.delta),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp_path = Path(tmp.name)
    try:
        scp_to_remote(conn, tmp_path, f"{rp.run_dir}/launch_manifest.json")
    finally:
        tmp_path.unlink(missing_ok=True)


def launch_batch(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.batch_manifest).expanduser().resolve())
    local = local_batch_paths(REPO_ROOT, manifest.batch_id)
    ssh_key = Path(args.ssh_key_path).expanduser().resolve()
    if not ssh_key.exists():
        raise SystemExit(f"SSH key not found: {ssh_key}")
    api = RunpodClient(api_key=args.runpod_api_key) if not args.dry_run else None

    inventory = _load_inventory(Path(args.pod_inventory).expanduser().resolve())
    pod_lookup = {pod.alias: pod for pod in manifest.pods}
    for alias, pod in inventory.items():
        pod_lookup[alias] = pod
    pod_template: dict[str, Any] = {}
    if getattr(args, "pod_template_file", ""):
        pod_template = read_json(Path(args.pod_template_file).expanduser().resolve())

    if manifest.launch_policy.one_active_experiment_per_pod:
        used: set[str] = set()
        for exp in manifest.experiments:
            if exp.pod_alias in used:
                raise SystemExit(f"pod_alias={exp.pod_alias} has multiple active experiments")
            used.add(exp.pod_alias)

    launch_results: list[dict[str, Any]] = []
    for exp in manifest.experiments:
        pod = pod_lookup.get(exp.pod_alias)
        if (not pod or not pod.pod_id) and getattr(args, "create_missing_pods", False):
            if args.dry_run:
                pod = PodSpec(pod_id=f"dryrun-{exp.pod_alias}", alias=exp.pod_alias, host="<dry-run-host>", port=22, ssh_user="root")
                pod_lookup[exp.pod_alias] = pod
            else:
                if api is None:
                    raise SystemExit("RunPod API is required to create missing pod")
                payload = dict(pod_template)
                payload["name"] = payload.get("name", f"runpod-{manifest.batch_id}-{exp.pod_alias}")[:120]
                created = api.create_pod(payload)
                pod = PodSpec(pod_id=str(created["id"]), alias=exp.pod_alias, ssh_user="root")
                pod_lookup[exp.pod_alias] = pod
        if not pod or not pod.pod_id:
            raise SystemExit(f"Missing pod assignment for alias={exp.pod_alias}")
        rp = remote_paths(manifest.batch_id, exp.exp_id, exp.run_dir)

        if args.dry_run:
            conn = PodConn(
                pod_id=pod.pod_id,
                host=pod.host or "<dry-run-host>",
                port=pod.port or 22,
                ssh_user=pod.ssh_user or "root",
                ssh_key_path=ssh_key,
            )
            result = LaunchResult(
                exp_id=exp.exp_id,
                pod_id=pod.pod_id,
                pod_alias=pod.alias,
                host=conn.host,
                port=conn.port,
                run_dir=rp.run_dir,
                release_dir=rp.release_dir,
                log_path=rp.log_path,
                tail_command=_tail_cmd(conn, rp.log_path),
                state_path=rp.state_path,
            )
            launch_results.append(asdict(result))
            continue

        if api is None:
            raise SystemExit("runpod api client is unavailable")
        conn = _resolve_conn(api, pod, ssh_key)
        if not getattr(args, "allow_reuse_run_dir", False):
            exists = run_ssh(
                conn,
                f"if [ -f {shlex.quote(rp.run_dir + '/run_state.json')} ]; then echo exists; fi",
                retries=2,
            )
            if "exists" in (exists.stdout or ""):
                raise SystemExit(
                    f"run_dir already has run_state.json for exp_id={exp.exp_id}. "
                    "Use relaunch_experiment or --allow-reuse-run-dir explicitly."
                )
        _deploy_release(conn, manifest, exp, rp)
        _write_remote_launch_manifest(conn, exp, rp, manifest)
        exp_command = _build_experiment_command(manifest, exp, rp)
        command_b64 = base64.b64encode(exp_command.encode("utf-8")).decode("ascii")
        launcher = (
            f"mkdir -p {shlex.quote(rp.run_dir)} && "
            f"nohup python3 {shlex.quote(rp.release_dir + '/scripts/runpod_jobs/remote_launcher.py')} "
            f"--exp-id {shlex.quote(exp.exp_id)} "
            f"--run-dir {shlex.quote(rp.run_dir)} "
            f"--log-path {shlex.quote(rp.log_path)} "
            f"--command-b64 {shlex.quote(command_b64)} "
            f"> {shlex.quote(rp.log_path)} 2>&1 < /dev/null & echo $!"
        )
        proc = run_ssh(conn, launcher)
        pid = int((proc.stdout or "0").strip().splitlines()[-1])
        run_ssh(conn, f"echo {pid} > {shlex.quote(rp.run_dir + '/pid.txt')}")

        result = LaunchResult(
            exp_id=exp.exp_id,
            pod_id=pod.pod_id,
            pod_alias=pod.alias,
            host=conn.host,
            port=conn.port,
            run_dir=rp.run_dir,
            release_dir=rp.release_dir,
            log_path=rp.log_path,
            tail_command=_tail_cmd(conn, rp.log_path),
            state_path=rp.state_path,
        )
        launch_results.append(asdict(result))
        if args.tail_only:
            print(result.tail_command)

    write_json(local.launch_results, {"batch_id": manifest.batch_id, "launch_results": launch_results})
    if not args.tail_only:
        print(str(local.launch_results))
    return 0


def _tail_cmd(conn: PodConn, log_path: str) -> str:
    return (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {conn.port} "
        f"-i \"{conn.ssh_key_path}\" {conn.ssh_user}@{conn.host} "
        f"\"tail -n 200 -f {log_path}\""
    )


def status_batch(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.batch_manifest).expanduser().resolve())
    local = local_batch_paths(REPO_ROOT, manifest.batch_id)
    launch_payload = read_json(local.launch_results)
    launch_items = list(launch_payload.get("launch_results", []))
    ssh_key = Path(args.ssh_key_path).expanduser().resolve()
    api = RunpodClient(api_key=args.runpod_api_key)
    status_items: list[dict[str, Any]] = []

    for item in launch_items:
        pod_id = str(item["pod_id"])
        endpoint = api.wait_for_ssh_endpoint(pod_id, timeout_seconds=60, poll_seconds=4)
        conn = PodConn(pod_id=pod_id, host=endpoint.host, port=endpoint.port, ssh_user="root", ssh_key_path=ssh_key)
        run_dir = str(item["run_dir"])
        state_data: dict[str, Any] = {}
        try:
            state_proc = run_ssh(conn, f"cat {shlex.quote(run_dir + '/run_state.json')}", retries=2)
            state_data = json.loads(state_proc.stdout or "{}")
        except Exception:
            state_data = {"state": RunStateName.UNKNOWN.value, "message": "state file unavailable"}
        artifact_ready = False
        try:
            m = run_ssh(conn, f"test -f {shlex.quote(run_dir + '/artifacts_manifest.json')} && echo ready || true", retries=2)
            artifact_ready = "ready" in (m.stdout or "")
        except Exception:
            artifact_ready = False
        status_items.append(
            {
                "exp_id": item["exp_id"],
                "pod_id": pod_id,
                "state": state_data.get("state", RunStateName.UNKNOWN.value),
                "started_at": state_data.get("started_at", ""),
                "last_heartbeat": state_data.get("last_heartbeat", ""),
                "pid": state_data.get("pid", 0),
                "exit_code": state_data.get("exit_code"),
                "remote_run_dir": run_dir,
                "log_path": item["log_path"],
                "artifact_ready": artifact_ready,
            }
        )
    write_json(local.status, {"batch_id": manifest.batch_id, "items": status_items, "updated_at": now_utc_iso()})
    print(str(local.status))
    return 0


def download_batch(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.batch_manifest).expanduser().resolve())
    local = local_batch_paths(REPO_ROOT, manifest.batch_id)
    launch_payload = read_json(local.launch_results)
    ssh_key = Path(args.ssh_key_path).expanduser().resolve()
    api = RunpodClient(api_key=args.runpod_api_key)
    include = manifest.artifact_policy.include_paths or [
        "run_report.md",
        "run_state.json",
        "launch_manifest.json",
        "artifacts_manifest.json",
        "pipeline.log",
        "summary.json",
    ]

    for item in launch_payload.get("launch_results", []):
        endpoint = api.wait_for_ssh_endpoint(str(item["pod_id"]), timeout_seconds=60, poll_seconds=4)
        conn = PodConn(
            pod_id=str(item["pod_id"]),
            host=endpoint.host,
            port=endpoint.port,
            ssh_user="root",
            ssh_key_path=ssh_key,
        )
        run_dir = str(item["run_dir"]).rstrip("/")
        exp_dir = ensure_dir(local.downloaded / str(item["exp_id"]))
        for rel in include:
            remote = f"{run_dir}/{rel}"
            try:
                scp_from_remote(conn, remote, exp_dir / Path(rel).name, recursive=False)
            except Exception:
                continue
    print(str(local.downloaded))
    return 0


def relaunch_experiment(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.batch_manifest).expanduser().resolve())
    target = args.exp_id
    experiments = [x for x in manifest.experiments if x.exp_id == target]
    if not experiments:
        raise SystemExit(f"exp_id not found: {target}")
    manifest.experiments = experiments
    tmp_manifest = Path(tempfile.mkstemp(prefix="runpod_relaunch_", suffix=".json")[1])
    try:
        write_json(tmp_manifest, _manifest_to_dict(manifest))
        launch_args = argparse.Namespace(
            batch_manifest=str(tmp_manifest),
            pod_inventory=args.pod_inventory,
            ssh_key_path=args.ssh_key_path,
            runpod_api_key=args.runpod_api_key,
            dry_run=args.dry_run,
            tail_only=args.tail_only,
            allow_reuse_run_dir=True,
            create_missing_pods=False,
            pod_template_file="",
        )
        return launch_batch(launch_args)
    finally:
        tmp_manifest.unlink(missing_ok=True)


def tail_command(args: argparse.Namespace) -> int:
    launch_payload = read_json(Path(args.launch_results).expanduser().resolve())
    item = next((x for x in launch_payload.get("launch_results", []) if str(x.get("exp_id")) == args.exp_id), None)
    if not item:
        raise SystemExit(f"exp_id not found in launch_results: {args.exp_id}")
    print(str(item["tail_command"]))
    return 0


def doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    ssh_key = Path(args.ssh_key_path).expanduser().resolve()
    checks.append({"check": "ssh_key_exists", "ok": ssh_key.exists(), "path": str(ssh_key)})
    api_ok = True
    err = ""
    try:
        api = RunpodClient(api_key=args.runpod_api_key)
        inventory = _load_inventory(Path(args.pod_inventory).expanduser().resolve())
        for alias, pod in inventory.items():
            endpoint = api.wait_for_ssh_endpoint(pod.pod_id, timeout_seconds=90, poll_seconds=5)
            conn = PodConn(pod_id=pod.pod_id, host=endpoint.host, port=endpoint.port, ssh_user=pod.ssh_user, ssh_key_path=ssh_key)
            checks.append({"check": f"ssh_ready:{alias}", "ok": check_ssh_ready(conn), "host": endpoint.host, "port": endpoint.port})
            try:
                df = run_ssh(conn, "df -h /tmp /workspace | sed -n '1,3p'")
                checks.append({"check": f"remote_space:{alias}", "ok": True, "details": (df.stdout or "").strip()})
            except Exception as exc:
                checks.append({"check": f"remote_space:{alias}", "ok": False, "error": str(exc)})
            try:
                py = run_ssh(conn, "python3 --version && (uv --version || true)")
                checks.append({"check": f"python_uv:{alias}", "ok": True, "details": (py.stdout or "").strip()})
            except Exception as exc:
                checks.append({"check": f"python_uv:{alias}", "ok": False, "error": str(exc)})
    except Exception as exc:
        api_ok = False
        err = str(exc)
    checks.append({"check": "runpod_api", "ok": api_ok, "error": err})
    print(json.dumps({"checks": checks}, ensure_ascii=False, indent=2))
    return 0
