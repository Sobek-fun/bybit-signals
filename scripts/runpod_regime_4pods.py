from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrapper for the generic batch launcher with default 4 experiments")
    parser.add_argument("--runpod-api-key", required=True)
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
    parser.add_argument("--base-run-dir", default="/workspace/experiments")
    parser.add_argument("--detector-dir", default="artifacts/tune_threshold_no_argmax_liq7d_detector")
    parser.add_argument("--tokens-file", default="config/regime_tokens_curated55.txt")
    parser.add_argument("--clickhouse-dsn", default="http://admin:GtChrHFvAL3CybQB@185.189.45.79:8123/bybit")
    parser.add_argument("--experiments", default="exp1,exp2,exp3,exp4")
    return parser.parse_args()


def default_specs(base_run_dir: str) -> list[dict[str, str]]:
    return [
        {"exp_id": "exp1", "run_root": f"{base_run_dir}/exp1_curated55_blockvalue_strict_notpsl"},
        {"exp_id": "exp2", "run_root": f"{base_run_dir}/exp2_curated55_local_features_blockvalue"},
        {"exp_id": "exp3", "run_root": f"{base_run_dir}/exp3_curated55_target_good_expanded_full"},
        {"exp_id": "exp4", "run_root": f"{base_run_dir}/exp4_curated55_combined_local_targetv3_min3"},
    ]


def main() -> None:
    args = parse_args()
    selected = {x.strip() for x in args.experiments.split(",") if x.strip()}
    specs = [x for x in default_specs(args.base_run_dir.rstrip("/")) if x["exp_id"] in selected]
    if not specs:
        raise SystemExit("No experiments selected")

    batch_script = Path(__file__).resolve().with_name("runpod_regime_batch.py")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        tmp.write(json.dumps(specs, ensure_ascii=False, indent=2))
        spec_path = Path(tmp.name)

    cmd = [
        "python",
        str(batch_script),
        "--runpod-api-key",
        args.runpod_api_key,
        "--spec-file",
        str(spec_path),
        "--storage-id",
        args.storage_id,
        "--storage-mount-path",
        args.storage_mount_path,
        "--image-name",
        args.image_name,
        "--ssh-user",
        args.ssh_user,
        "--ssh-key-path",
        args.ssh_key_path,
        "--project-root",
        args.project_root,
        "--remote-project-base",
        args.remote_project_base,
        "--container-disk-gb",
        str(args.container_disk_gb),
        "--create-retries",
        str(args.create_retries),
        "--create-retry-sleep-seconds",
        str(args.create_retry_sleep_seconds),
        "--detector-dir",
        args.detector_dir,
        "--tokens-file",
        args.tokens_file,
        "--clickhouse-dsn",
        args.clickhouse_dsn,
    ]
    proc = subprocess.run(cmd, check=False)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
