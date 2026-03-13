from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scripts.runpod_jobs.cli import main as orchestrator_main
from scripts.runpod_jobs.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated wrapper: start single experiment on existing pod")
    parser.add_argument("--runpod-api-key", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--ssh-key-path", default=str(Path("~/.ssh/id_ed25519").expanduser()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
    item = next((x for x in specs if str(x.get("exp_id")) == args.exp_id), None)
    if not item:
        raise SystemExit(f"exp_id not found in spec: {args.exp_id}")
    batch_id = f"legacy_single_{args.exp_id}"
    tmp = Path(tempfile.mkdtemp(prefix="runpod_legacy_single_"))
    manifest = {
        "batch_id": batch_id,
        "baseline": {"baseline_id": "legacy"},
        "shared_inputs": {},
        "pods": [{"pod_id": args.pod_id, "alias": "pod1"}],
        "experiments": [
            {
                "exp_id": args.exp_id,
                "pod_alias": "pod1",
                "run_dir": str(item.get("run_root", f"/workspace/experiments/{batch_id}/{args.exp_id}")),
                "delta": {"transform_script": str(item.get("transform_script", "scripts/runpod_jobs/apply_experiment_transform.py"))},
            }
        ],
    }
    pod_inventory = {"pods": [{"pod_id": args.pod_id, "alias": "pod1"}]}
    spec_file = tmp / "legacy_spec.json"
    manifest_file = tmp / "batch_manifest.json"
    pods_file = tmp / "pod_inventory.json"
    write_json(spec_file, manifest)
    write_json(pods_file, pod_inventory)

    import sys

    sys.argv = ["cli.py", "prepare_batch", "--spec-file", str(spec_file), "--batch-id", batch_id]
    orchestrator_main()
    generated_manifest = Path("artifacts") / "runpod_batches" / batch_id / "batch_manifest.json"
    if generated_manifest.exists():
        manifest_file = generated_manifest.resolve()
    sys.argv = [
        "cli.py",
        "launch_batch",
        "--batch-manifest",
        str(manifest_file),
        "--pod-inventory",
        str(pods_file),
        "--runpod-api-key",
        args.runpod_api_key,
        "--ssh-key-path",
        args.ssh_key_path,
        "--tail-only",
    ]
    orchestrator_main()


if __name__ == "__main__":
    main()
