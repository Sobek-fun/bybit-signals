from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.runpod_jobs.core import (
    doctor,
    download_batch,
    launch_batch,
    prepare_batch,
    relaunch_experiment,
    status_batch,
    tail_command,
)


def _default_ssh_key() -> str:
    return str(Path("~/.ssh/id_ed25519").expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod batch orchestrator (baseline + isolated deltas)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare_batch")
    p_prepare.add_argument("--spec-file", required=True, help="input experiment spec json")
    p_prepare.add_argument("--batch-id", default="", help="optional explicit batch id")

    p_launch = sub.add_parser("launch_batch")
    p_launch.add_argument("--batch-manifest", required=True)
    p_launch.add_argument("--pod-inventory", required=True)
    p_launch.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_launch.add_argument("--ssh-key-path", default=_default_ssh_key())
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.add_argument("--tail-only", action="store_true")
    p_launch.add_argument("--allow-reuse-run-dir", action="store_true")
    p_launch.add_argument("--create-missing-pods", action="store_true")
    p_launch.add_argument("--pod-template-file", default="", help="json template for RunPod create payload")

    p_status = sub.add_parser("status_batch")
    p_status.add_argument("--batch-manifest", required=True)
    p_status.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_status.add_argument("--ssh-key-path", default=_default_ssh_key())

    p_download = sub.add_parser("download_batch")
    p_download.add_argument("--batch-manifest", required=True)
    p_download.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_download.add_argument("--ssh-key-path", default=_default_ssh_key())

    p_relaunch = sub.add_parser("relaunch_experiment")
    p_relaunch.add_argument("--batch-manifest", required=True)
    p_relaunch.add_argument("--pod-inventory", required=True)
    p_relaunch.add_argument("--exp-id", required=True)
    p_relaunch.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_relaunch.add_argument("--ssh-key-path", default=_default_ssh_key())
    p_relaunch.add_argument("--dry-run", action="store_true")
    p_relaunch.add_argument("--tail-only", action="store_true")

    p_tail = sub.add_parser("tail_command")
    p_tail.add_argument("--launch-results", required=True)
    p_tail.add_argument("--exp-id", required=True)

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--pod-inventory", required=True)
    p_doctor.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_doctor.add_argument("--ssh-key-path", default=_default_ssh_key())

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare_batch":
        raise SystemExit(prepare_batch(args))
    if args.command == "launch_batch":
        raise SystemExit(launch_batch(args))
    if args.command == "status_batch":
        raise SystemExit(status_batch(args))
    if args.command == "download_batch":
        raise SystemExit(download_batch(args))
    if args.command == "relaunch_experiment":
        raise SystemExit(relaunch_experiment(args))
    if args.command == "tail_command":
        raise SystemExit(tail_command(args))
    if args.command == "doctor":
        raise SystemExit(doctor(args))
    raise SystemExit(parser.format_help())


if __name__ == "__main__":
    main()
