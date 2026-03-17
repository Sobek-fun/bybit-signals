from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.runpod_jobs.models import BatchSpec, ExperimentSpec, LaunchResult, PodSpec
from scripts.runpod_jobs.paths import SHARED_VENV_DIR, local_paths, remote_paths
from scripts.runpod_jobs.runpod_api import RunpodClient
from scripts.runpod_jobs.ssh import PodConn, check_ssh_ready, run_ssh, scp_to_remote
from scripts.runpod_jobs.utils import ensure_dir, read_json, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIRS = ("pump_end_threshold", "pump_end_prod", "scripts")


def _log(stage: str, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[runpod_jobs][{ts}][{stage}] {message}")


def _load_spec(path: Path) -> BatchSpec:
    payload = read_json(path)
    spec = BatchSpec.from_dict(payload)
    _validate_spec(spec)
    return spec


def _load_inventory(path: Path) -> dict[str, PodSpec]:
    payload = read_json(path)
    items = payload.get("pods", []) if isinstance(payload, dict) else payload
    out: dict[str, PodSpec] = {}
    for item in items:
        pod = PodSpec.from_dict(item)
        if not pod.alias:
            continue
        if pod.alias in out:
            raise SystemExit(f"duplicate pod alias in inventory: {pod.alias}")
        out[pod.alias] = pod
    return out


def _validate_spec(spec: BatchSpec) -> None:
    if not spec.batch_id:
        raise SystemExit("spec.batch_id is required")
    if not spec.experiments:
        raise SystemExit("spec.experiments must be non-empty")
    if not spec.runtime.pipeline_command:
        raise SystemExit("spec.runtime.pipeline_command is required")
    if not spec.runtime.requirements_file:
        raise SystemExit("spec.runtime.requirements_file is required")
    _assert_allowed_relative_path(spec.runtime.requirements_file, "spec.runtime.requirements_file")
    exp_ids = [exp.exp_id for exp in spec.experiments]
    if len(exp_ids) != len(set(exp_ids)):
        raise SystemExit("exp_id must be unique")
    for exp in spec.experiments:
        if not exp.exp_id:
            raise SystemExit("experiment.exp_id is required")
        if not exp.pod_alias:
            raise SystemExit(f"{exp.exp_id}: pod_alias is required")
        if len(exp.patch_files) > 1:
            raise SystemExit(
                f"{exp.exp_id}: only one patch file is allowed per experiment; got {len(exp.patch_files)}"
            )


def _assert_allowed_relative_path(rel: str, field_name: str) -> None:
    p = Path(rel.replace("\\", "/"))
    if p.is_absolute():
        raise SystemExit(f"{field_name} must be relative to repository root")
    if ".." in p.parts:
        raise SystemExit(f"{field_name} cannot contain '..': {rel}")
    if not p.parts:
        raise SystemExit(f"{field_name} is empty")
    if p.parts[0] not in ALLOWED_DIRS:
        raise SystemExit(f"{field_name} must be under {ALLOWED_DIRS}: {rel}")


def _assert_repo_relative_path(rel: str, field_name: str) -> None:
    p = Path(rel.replace("\\", "/"))
    if p.is_absolute():
        raise SystemExit(f"{field_name} must be relative to repository root")
    if ".." in p.parts or not p.parts:
        raise SystemExit(f"{field_name} contains invalid path: {rel}")


def _run_local(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, cwd=str(cwd) if cwd else None, check=False)


def _validate_patch_allowed_paths(patch_path: Path) -> None:
    lines = patch_path.read_text(encoding="utf-8", errors="replace").splitlines()
    touched: set[str] = set()
    for line in lines:
        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            normalized = raw.replace("\\", "/")
            rel = Path(normalized)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise SystemExit(f"patch path is not allowed in {patch_path}: {raw}")
            if rel.parts[0] not in ALLOWED_DIRS:
                raise SystemExit(f"patch touches disallowed path in {patch_path}: {raw}")
            touched.add(rel.parts[0])
    if not touched:
        raise SystemExit(f"patch has no file targets: {patch_path}")


def _copy_baseline_subset(dst_src: Path) -> None:
    ensure_dir(dst_src)
    for name in ALLOWED_DIRS:
        src = REPO_ROOT / name
        if not src.exists():
            raise SystemExit(f"required baseline directory is missing: {src}")
        dst = dst_src / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, dirs_exist_ok=False)


def _assert_snapshot_contains_only_allowed(src_root: Path) -> None:
    children = [p.name for p in src_root.iterdir()]
    extras = sorted(x for x in children if x not in ALLOWED_DIRS)
    if extras:
        raise SystemExit(f"snapshot contains disallowed top-level dirs: {extras}")


def _assemble_experiment(exp: ExperimentSpec, lp: Any) -> Path:
    exp_root = lp.assembled_root / exp.exp_id
    src_root = exp_root / "src"
    shutil.rmtree(exp_root, ignore_errors=True)
    ensure_dir(exp_root)
    _log("assemble", f"{exp.exp_id}: copy baseline subset")
    _copy_baseline_subset(src_root)
    for rel_patch in exp.patch_files:
        _log("assemble", f"{exp.exp_id}: apply patch {rel_patch}")
        _assert_repo_relative_path(rel_patch, f"{exp.exp_id}.patch_files[]")
        patch_path = (REPO_ROOT / rel_patch).resolve()
        if not patch_path.exists():
            raise SystemExit(f"patch file not found for {exp.exp_id}: {patch_path}")
        _validate_patch_allowed_paths(patch_path)
        check = _run_local(["git", "-C", str(src_root), "apply", "--check", str(patch_path)])
        if check.returncode != 0:
            details = (check.stderr or check.stdout or "").strip()
            raise SystemExit(f"patch check failed for {exp.exp_id}: {patch_path}\n{details}")
        apply_proc = _run_local(["git", "-C", str(src_root), "apply", str(patch_path)])
        if apply_proc.returncode != 0:
            details = (apply_proc.stderr or apply_proc.stdout or "").strip()
            raise SystemExit(f"patch apply failed for {exp.exp_id}: {patch_path}\n{details}")
    _assert_snapshot_contains_only_allowed(src_root)
    return src_root


def _ensure_shared_venv(
    spec: BatchSpec,
    experiments: list[ExperimentSpec],
    inventory: dict[str, PodSpec],
    ssh_key: Path,
    api: RunpodClient | None,
    dry_run: bool,
) -> None:
    if dry_run or not experiments:
        return
    requirements_path = (REPO_ROOT / spec.runtime.requirements_file).resolve()
    if not requirements_path.exists():
        raise SystemExit(f"requirements file not found: {requirements_path}")
    bootstrap_exp = experiments[0]
    bootstrap_pod = inventory[bootstrap_exp.pod_alias]
    _log("venv", f"bootstrap shared venv on pod alias={bootstrap_exp.pod_alias}")
    conn = _resolve_conn(bootstrap_pod, ssh_key, api)
    remote_requirements = f"/tmp/runpod_requirements_{spec.batch_id}.txt".replace(" ", "_")
    scp_to_remote(conn, requirements_path, remote_requirements)
    cmd = (
        "set -euo pipefail; "
        f"VENV={shlex.quote(SHARED_VENV_DIR)}; "
        f"REQ={shlex.quote(remote_requirements)}; "
        "MARKER=\"$VENV/.ready\"; "
        "LOCK=\"$VENV.lock\"; "
        "if [[ -x \"$VENV/bin/python\" && -f \"$MARKER\" ]]; then "
        "  echo 'shared_venv_ready'; "
        "  exit 0; "
        "fi; "
        "mkdir -p \"$(dirname \"$VENV\")\"; "
        "while ! mkdir \"$LOCK\" 2>/dev/null; do sleep 1; done; "
        "trap 'rmdir \"$LOCK\" 2>/dev/null || true' EXIT; "
        "if [[ ! -x \"$VENV/bin/python\" ]]; then "
        "  rm -rf \"$VENV\"; "
        f"  {shlex.quote(spec.runtime.python_bin)} -m venv \"$VENV\"; "
        "fi; "
        "source \"$VENV/bin/activate\"; "
        "if ! python -m pip --version >/dev/null 2>&1; then python -m ensurepip --upgrade; fi; "
        "python -m pip install -r \"$REQ\"; "
        "touch \"$MARKER\"; "
        "echo 'shared_venv_bootstrap_done'; "
    )
    run_ssh(conn, f"bash -lc {shlex.quote(cmd)}", timeout=3600)
    _log("venv", f"shared venv ready at {SHARED_VENV_DIR}")


def _resolve_conn(pod: PodSpec, ssh_key: Path, api: RunpodClient | None) -> PodConn:
    if pod.host:
        conn = PodConn(
            pod_id=pod.pod_id or pod.alias,
            host=pod.host,
            port=pod.port or 22,
            ssh_user=pod.ssh_user or "root",
            ssh_key_path=ssh_key,
        )
    else:
        if not pod.pod_id:
            raise SystemExit(f"pod {pod.alias} must have either host or pod_id")
        if api is None:
            raise SystemExit(f"pod {pod.alias} requires RunPod API key to resolve SSH endpoint")
        endpoint = api.wait_for_ssh_endpoint(pod.pod_id, timeout_seconds=600, poll_seconds=6)
        conn = PodConn(
            pod_id=pod.pod_id,
            host=endpoint.host,
            port=endpoint.port,
            ssh_user=pod.ssh_user or "root",
            ssh_key_path=ssh_key,
        )
    if not check_ssh_ready(conn):
        raise SystemExit(f"ssh is not ready for pod alias={pod.alias}")
    return conn


def _build_launch_command(spec: BatchSpec, exp: ExperimentSpec, rp: Any) -> str:
    env_parts = [
        f"RUN_ROOT={shlex.quote(rp.run_dir)}",
        f"DETECTOR_DIR={shlex.quote(spec.runtime.detector_dir_remote)}",
        f"TOKENS_FILE={shlex.quote(spec.runtime.tokens_file_remote)}",
        f"EXP_ID={shlex.quote(exp.exp_id)}",
        f"BATCH_ID={shlex.quote(spec.batch_id)}",
    ]
    return " ".join(env_parts + ["bash", "-lc", shlex.quote(spec.runtime.pipeline_command)])


def _tail_command(conn: PodConn, rp: Any) -> str:
    return (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {conn.port} "
        f"-i \"{conn.ssh_key_path}\" {conn.ssh_user}@{conn.host} "
        f"\"tail -n 200 -f {rp.log_path}\""
    )


def _start_script(spec: BatchSpec, rp: Any, launch_command: str) -> str:
    setup_line = spec.runtime.setup_command.strip()
    export_lines: list[str] = []
    for key, value in sorted(spec.runtime.extra_env.items()):
        rendered = value.replace("{run_root}", rp.run_dir)
        export_lines.append(f"export {key}={shlex.quote(rendered)}")
    exports_block = "\n".join(export_lines)
    if exports_block:
        exports_block += "\n"
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUNNER={shlex.quote(rp.src_dir + "/scripts/runpod_jobs/run_experiment.sh")}
if [[ ! -x "$RUNNER" ]]; then
  chmod +x "$RUNNER"
fi
{exports_block}bash "$RUNNER" \\
  --src-dir {shlex.quote(rp.src_dir)} \\
  --run-dir {shlex.quote(rp.run_dir)} \\
  --venv-dir {shlex.quote(rp.venv_dir)} \\
  --log-path {shlex.quote(rp.log_path)} \\
  --started-at-path {shlex.quote(rp.started_at_path)} \\
  --finished-at-path {shlex.quote(rp.finished_at_path)} \\
  --exit-code-path {shlex.quote(rp.exit_code_path)} \\
  --launch-command-path {shlex.quote(rp.launch_command_path)} \\
  --python-bin {shlex.quote(spec.runtime.python_bin)} \\
  --requirements-file {shlex.quote(spec.runtime.requirements_file.replace("\\\\", "/"))} \\
  --clickhouse-dsn-env {shlex.quote(spec.runtime.clickhouse_dsn_env)} \\
  --detector-dir-remote {shlex.quote(spec.runtime.detector_dir_remote)} \\
  --tokens-file-remote {shlex.quote(spec.runtime.tokens_file_remote)} \\
  --setup-command {shlex.quote(setup_line)} \\
  --launch-command {shlex.quote(launch_command)}
"""


def _deploy_and_start(
    spec: BatchSpec,
    exp: ExperimentSpec,
    src_root: Path,
    conn: PodConn,
    relaunch_mode: bool,
    dry_run: bool,
) -> LaunchResult:
    rp = remote_paths(spec.runtime.workspace_root, spec.batch_id, exp.exp_id)
    launch_command = _build_launch_command(spec, exp, rp)
    if dry_run:
        return LaunchResult(
            exp_id=exp.exp_id,
            pod_alias=exp.pod_alias,
            pod_id=conn.pod_id,
            host=conn.host,
            port=conn.port,
            exp_root=rp.exp_root,
            tail_command=_tail_command(conn, rp),
            launch_command=launch_command,
        )

    _log("deploy", f"{exp.exp_id}: prepare remote dir {rp.exp_root}")
    if relaunch_mode:
        run_ssh(conn, f"rm -rf {shlex.quote(rp.exp_root)}")
    else:
        exists = run_ssh(conn, f"if [ -d {shlex.quote(rp.exp_root)} ]; then echo exists; fi")
        if "exists" in (exists.stdout or ""):
            raise SystemExit(
                f"remote experiment dir already exists for {exp.exp_id}: {rp.exp_root}. "
                f"Use relaunch command for restart."
            )

    run_ssh(conn, f"mkdir -p {shlex.quote(rp.exp_root)}")
    _log("deploy", f"{exp.exp_id}: upload snapshot")
    scp_to_remote(conn, src_root, rp.exp_root, recursive=True)

    _log("deploy", f"{exp.exp_id}: render and upload starter script")
    script_text = _start_script(spec, rp, launch_command)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8", newline="\n") as tmp:
        tmp.write(script_text.replace("\r\n", "\n"))
        tmp_path = Path(tmp.name)
    try:
        scp_to_remote(conn, tmp_path, rp.start_script_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    run_ssh(conn, f"chmod +x {shlex.quote(rp.start_script_path)}")
    _log("deploy", f"{exp.exp_id}: start pipeline in background")
    run_ssh(
        conn,
        f"nohup bash {shlex.quote(rp.start_script_path)} > {shlex.quote(rp.log_path)} 2>&1 < /dev/null & echo $!",
    )
    return LaunchResult(
        exp_id=exp.exp_id,
        pod_alias=exp.pod_alias,
        pod_id=conn.pod_id,
        host=conn.host,
        port=conn.port,
        exp_root=rp.exp_root,
        tail_command=_tail_command(conn, rp),
        launch_command=launch_command,
    )


def _launch_impl(args: argparse.Namespace, relaunch_mode: bool) -> int:
    _log("start", f"command={'relaunch' if relaunch_mode else 'launch'}")
    spec = _load_spec(Path(args.spec_file).expanduser().resolve())
    lp = local_paths(REPO_ROOT, spec.batch_id)
    ensure_dir(lp.root)
    ensure_dir(lp.assembled_root)

    ssh_key = Path(args.ssh_key_path).expanduser().resolve()
    if not ssh_key.exists():
        raise SystemExit(f"ssh key not found: {ssh_key}")

    inventory = _load_inventory(Path(args.pod_inventory).expanduser().resolve())
    for exp in spec.experiments:
        if exp.pod_alias not in inventory:
            raise SystemExit(f"{exp.exp_id}: pod_alias is missing in inventory: {exp.pod_alias}")

    target_exp_id = getattr(args, "exp_id", "")
    experiments = spec.experiments
    if target_exp_id:
        experiments = [x for x in experiments if x.exp_id == target_exp_id]
        if not experiments:
            raise SystemExit(f"exp_id not found in spec: {target_exp_id}")
    spec.experiments = experiments

    _log("assemble", f"build local snapshots count={len(spec.experiments)}")
    assembled: dict[str, Path] = {}
    for exp in spec.experiments:
        assembled[exp.exp_id] = _assemble_experiment(exp, lp)

    api: RunpodClient | None = None
    needs_api = any(not inventory[exp.pod_alias].host for exp in spec.experiments)
    if needs_api and not args.dry_run:
        api = RunpodClient(api_key=args.runpod_api_key)

    _ensure_shared_venv(spec, spec.experiments, inventory, ssh_key, api, args.dry_run)

    launch_commands = {exp.exp_id: _build_launch_command(spec, exp, remote_paths(spec.runtime.workspace_root, spec.batch_id, exp.exp_id)) for exp in spec.experiments}
    if len(set(launch_commands.values())) != len(launch_commands):
        raise SystemExit("launch commands are not unique per experiment")

    results: list[LaunchResult] = []
    failures: list[str] = []

    def _one(exp: ExperimentSpec) -> LaunchResult:
        pod = inventory[exp.pod_alias]
        if args.dry_run:
            dry_conn = PodConn(
                pod_id=pod.pod_id or pod.alias,
                host=pod.host or "<dry-run-host>",
                port=pod.port or 22,
                ssh_user=pod.ssh_user,
                ssh_key_path=ssh_key,
            )
            return _deploy_and_start(spec, exp, assembled[exp.exp_id], dry_conn, relaunch_mode, dry_run=True)
        conn = _resolve_conn(pod, ssh_key, api)
        return _deploy_and_start(spec, exp, assembled[exp.exp_id], conn, relaunch_mode, dry_run=False)

    max_parallel = max(1, int(getattr(args, "max_parallel", 4)))
    _log("launch", f"starting deployment max_parallel={max_parallel}")
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        future_map = {ex.submit(_one, exp): exp for exp in spec.experiments}
        for future in as_completed(future_map):
            exp = future_map[future]
            try:
                results.append(future.result())
                _log("launch", f"{exp.exp_id}: started")
            except BaseException as exc:
                failures.append(f"{exp.exp_id}: {exc}")
                _log("launch", f"{exp.exp_id}: failed {exc}")

    payload = {
        "batch_id": spec.batch_id,
        "results": [asdict(x) for x in sorted(results, key=lambda y: y.exp_id)],
        "failures": failures,
    }
    write_json(lp.launch_results, payload)

    for item in sorted(results, key=lambda x: x.exp_id):
        print(item.tail_command)

    if failures:
        for message in failures:
            print(message, file=sys.stderr)
        return 1
    return 0


def launch(args: argparse.Namespace) -> int:
    return _launch_impl(args, relaunch_mode=False)


def relaunch(args: argparse.Namespace) -> int:
    return _launch_impl(args, relaunch_mode=True)


def doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    ok = True

    spec = _load_spec(Path(args.spec_file).expanduser().resolve())
    inventory = _load_inventory(Path(args.pod_inventory).expanduser().resolve())
    ssh_key = Path(args.ssh_key_path).expanduser().resolve()

    checks.append({"check": "ssh_key_exists", "ok": ssh_key.exists(), "path": str(ssh_key)})
    if not ssh_key.exists():
        ok = False

    for name in ALLOWED_DIRS:
        exists = (REPO_ROOT / name).exists()
        checks.append({"check": f"local_dir:{name}", "ok": exists})
        ok = ok and exists

    try:
        lp = local_paths(REPO_ROOT, spec.batch_id)
        ensure_dir(lp.root)
        ensure_dir(lp.assembled_root)
        for exp in spec.experiments:
            if exp.pod_alias not in inventory:
                raise SystemExit(f"{exp.exp_id}: pod_alias missing in inventory")
            _assemble_experiment(exp, lp)
        checks.append({"check": "local_assembly", "ok": True})
    except BaseException as exc:
        checks.append({"check": "local_assembly", "ok": False, "error": str(exc)})
        ok = False

    try:
        cmds = [_build_launch_command(spec, exp, remote_paths(spec.runtime.workspace_root, spec.batch_id, exp.exp_id)) for exp in spec.experiments]
        unique = len(cmds) == len(set(cmds))
        checks.append({"check": "unique_launch_commands", "ok": unique})
        ok = ok and unique
    except BaseException as exc:
        checks.append({"check": "unique_launch_commands", "ok": False, "error": str(exc)})
        ok = False

    api: RunpodClient | None = None
    needs_api = any(not inventory[exp.pod_alias].host for exp in spec.experiments)
    if needs_api:
        try:
            api = RunpodClient(api_key=args.runpod_api_key)
            checks.append({"check": "runpod_api", "ok": True})
        except BaseException as exc:
            checks.append({"check": "runpod_api", "ok": False, "error": str(exc)})
            ok = False

    seen_aliases: set[str] = set()
    for exp in spec.experiments:
        if exp.pod_alias in seen_aliases:
            continue
        seen_aliases.add(exp.pod_alias)
        pod = inventory.get(exp.pod_alias)
        if pod is None:
            checks.append({"check": f"pod:{exp.pod_alias}", "ok": False, "error": "missing in inventory"})
            ok = False
            continue
        try:
            conn = _resolve_conn(pod, ssh_key, api)
            checks.append({"check": f"ssh_ready:{exp.pod_alias}", "ok": True, "host": conn.host, "port": conn.port})
            py = run_ssh(conn, "python3 --version")
            checks.append({"check": f"python:{exp.pod_alias}", "ok": True, "details": (py.stdout or "").strip()})
            df = run_ssh(conn, "df -h /workspace | sed -n '1,2p'")
            checks.append({"check": f"workspace_space:{exp.pod_alias}", "ok": True, "details": (df.stdout or "").strip()})
            det = run_ssh(
                conn,
                f"if [ -d {shlex.quote(spec.runtime.detector_dir_remote)} ]; then echo ok; else echo missing; fi",
            ) if spec.runtime.detector_dir_remote else None
            if spec.runtime.detector_dir_remote:
                det_ok = "ok" in (det.stdout or "")
                checks.append({"check": f"detector_dir:{exp.pod_alias}", "ok": det_ok})
                ok = ok and det_ok
            tok = run_ssh(
                conn,
                f"if [ -f {shlex.quote(spec.runtime.tokens_file_remote)} ]; then echo ok; else echo missing; fi",
            ) if spec.runtime.tokens_file_remote else None
            if spec.runtime.tokens_file_remote:
                tok_ok = "ok" in (tok.stdout or "")
                checks.append({"check": f"tokens_file:{exp.pod_alias}", "ok": tok_ok})
                ok = ok and tok_ok
            dsn_env = spec.runtime.clickhouse_dsn_env
            if dsn_env:
                if spec.runtime.extra_env.get(dsn_env, "").strip():
                    dsn_ok = True
                else:
                    env_check = run_ssh(conn, f"if [ -n \"${{{dsn_env}:-}}\" ]; then echo ok; else echo missing; fi")
                    dsn_ok = "ok" in (env_check.stdout or "")
                checks.append({"check": f"dsn_env:{exp.pod_alias}", "ok": dsn_ok, "env": dsn_env})
                ok = ok and dsn_ok
        except BaseException as exc:
            checks.append({"check": f"pod_health:{exp.pod_alias}", "ok": False, "error": str(exc)})
            ok = False

    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if ok else 1
