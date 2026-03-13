from __future__ import annotations

import argparse
import base64
import json
import os
import select
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_state(
    *,
    run_dir: Path,
    exp_id: str,
    state: str,
    started_at: str,
    pid: int,
    exit_code: int | None,
    message: str = "",
) -> None:
    payload = {
        "exp_id": exp_id,
        "state": state,
        "started_at": started_at,
        "finished_at": now_utc() if state in {"FINISHED", "FAILED"} else "",
        "last_heartbeat": now_utc(),
        "pid": pid,
        "exit_code": exit_code,
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "pipeline.log"),
        "message": message,
    }
    write_text(run_dir / "run_state.json", json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote detached launcher with run_state machine")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--command-b64", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()
    pid = os.getpid()
    write_text(run_dir / "started_at.txt", started_at)
    write_text(run_dir / "pid.txt", str(pid))
    write_state(run_dir=run_dir, exp_id=args.exp_id, state="PREPARING", started_at=started_at, pid=pid, exit_code=None)

    command = base64.b64decode(args.command_b64.encode("ascii")).decode("utf-8")
    write_text(run_dir / "launcher_command.txt", command)

    write_state(run_dir=run_dir, exp_id=args.exp_id, state="DEPLOYING", started_at=started_at, pid=pid, exit_code=None)
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    write_state(run_dir=run_dir, exp_id=args.exp_id, state="BOOTSTRAPPING", started_at=started_at, pid=pid, exit_code=None)

    log_file = Path(args.log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        write_state(run_dir=run_dir, exp_id=args.exp_id, state="RUNNING", started_at=started_at, pid=pid, exit_code=None)
        while True:
            line = ""
            if proc.stdout is not None:
                ready, _, _ = select.select([proc.stdout], [], [], 5.0)
                if ready:
                    line = proc.stdout.readline()
            if line:
                log.write(line)
                log.flush()
            if proc.poll() is not None:
                if proc.stdout is not None:
                    remainder = proc.stdout.read()
                    if remainder:
                        log.write(remainder)
                        log.flush()
                break
            write_state(run_dir=run_dir, exp_id=args.exp_id, state="RUNNING", started_at=started_at, pid=pid, exit_code=None)

    rc = int(proc.returncode or 0)
    write_text(run_dir / "exit_code.txt", str(rc))
    write_text(run_dir / "finished_at.txt", now_utc())
    if rc == 0:
        write_state(run_dir=run_dir, exp_id=args.exp_id, state="FINISHED", started_at=started_at, pid=pid, exit_code=0)
        write_text(run_dir / "done.marker", "ok\n")
    else:
        write_state(run_dir=run_dir, exp_id=args.exp_id, state="FAILED", started_at=started_at, pid=pid, exit_code=rc)
        write_text(run_dir / "failed.marker", f"{rc}\n")
    raise SystemExit(rc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise SystemExit(130)
