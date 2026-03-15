from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


TRANSIENT_ERRORS = (
    "Connection reset",
    "Operation timed out",
    "Connection timed out",
    "Connection refused",
    "No route to host",
    "EOF",
    "Broken pipe",
)


class SshError(RuntimeError):
    pass


@dataclass(slots=True)
class PodConn:
    pod_id: str
    host: str
    port: int
    ssh_user: str
    ssh_key_path: Path


def _is_transient(output: str) -> bool:
    return any(item.lower() in output.lower() for item in TRANSIENT_ERRORS)


def _run_local(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
    if check and proc.returncode != 0:
        output = (proc.stderr or proc.stdout or "").strip()
        raise SshError(output or f"failed: {' '.join(cmd)}")
    return proc


def ssh_base(conn: PodConn) -> list[str]:
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


def scp_base(conn: PodConn) -> list[str]:
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


def run_ssh(conn: PodConn, command: str, retries: int = 4, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    delay = 2
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            proc = _run_local(ssh_base(conn) + [command], check=False, timeout=timeout)
            if proc.returncode == 0:
                return proc
            output = (proc.stderr or proc.stdout or "").strip()
            if attempt < retries and _is_transient(output):
                time.sleep(delay)
                delay *= 2
                continue
            raise SshError(output or "ssh command failed")
        except (subprocess.TimeoutExpired, SshError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            if _is_transient(str(exc)):
                time.sleep(delay)
                delay *= 2
                continue
            break
    raise SshError(str(last_exc or "ssh failed"))


def scp_to_remote(
    conn: PodConn,
    local_path: Path,
    remote_path: str,
    retries: int = 4,
    recursive: bool = False,
) -> None:
    delay = 2
    for attempt in range(1, retries + 1):
        cmd = scp_base(conn)
        if recursive:
            cmd.append("-r")
        cmd.extend([str(local_path), f"{conn.ssh_user}@{conn.host}:{remote_path}"])
        proc = _run_local(cmd, check=False)
        if proc.returncode == 0:
            return
        output = (proc.stderr or proc.stdout or "").strip()
        if attempt < retries and _is_transient(output):
            time.sleep(delay)
            delay *= 2
            continue
        raise SshError(output or "scp_to_remote failed")


def scp_from_remote(conn: PodConn, remote_path: str, local_path: Path, recursive: bool = True, retries: int = 4) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = scp_base(conn)
    if recursive:
        cmd.append("-r")
    cmd.extend([f"{conn.ssh_user}@{conn.host}:{remote_path}", str(local_path)])
    delay = 2
    for attempt in range(1, retries + 1):
        proc = _run_local(cmd, check=False)
        if proc.returncode == 0:
            return
        output = (proc.stderr or proc.stdout or "").strip()
        if attempt < retries and _is_transient(output):
            time.sleep(delay)
            delay *= 2
            continue
        raise SshError(output or "scp_from_remote failed")


def check_ssh_ready(conn: PodConn) -> bool:
    try:
        proc = _run_local(ssh_base(conn) + ["true"], check=False, timeout=20)
    except Exception:
        return False
    return proc.returncode == 0
