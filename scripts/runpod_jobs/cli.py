from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.runpod_jobs.core import doctor, launch, relaunch


def _default_ssh_key() -> str:
    return str(Path("~/.ssh/id_ed25519").expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RunPod launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--spec-file", required=True)
    p_launch.add_argument("--pod-inventory", required=True)
    p_launch.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_launch.add_argument("--ssh-key-path", default=_default_ssh_key())
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.add_argument("--max-parallel", type=int, default=4)

    p_relaunch = sub.add_parser("relaunch")
    p_relaunch.add_argument("--spec-file", required=True)
    p_relaunch.add_argument("--pod-inventory", required=True)
    p_relaunch.add_argument("--exp-id", required=True)
    p_relaunch.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_relaunch.add_argument("--ssh-key-path", default=_default_ssh_key())
    p_relaunch.add_argument("--dry-run", action="store_true")
    p_relaunch.add_argument("--max-parallel", type=int, default=1)

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--spec-file", required=True)
    p_doctor.add_argument("--pod-inventory", required=True)
    p_doctor.add_argument("--runpod-api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p_doctor.add_argument("--ssh-key-path", default=_default_ssh_key())

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "launch":
        raise SystemExit(launch(args))
    if args.command == "relaunch":
        raise SystemExit(relaunch(args))
    if args.command == "doctor":
        raise SystemExit(doctor(args))
    raise SystemExit(parser.format_help())


if __name__ == "__main__":
    main()
