import argparse
import time
from pathlib import Path

from pump_end_v2.artifacts import ArtifactManager
from pump_end_v2.config import REQUIRED_SECTIONS, load_and_validate_config
from pump_end_v2.logging import log_info, stage_done, stage_start
from pump_end_v2.pipeline import run_pump_end_v2_pipeline
from pump_end_v2.run_context import create_run_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pump_end_v2 foundation dry-run")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Run foundation dry-run")
    parser.add_argument(
        "--clickhouse-dsn",
        default=None,
        help="ClickHouse DSN for full-run mode",
    )
    return parser


def run_dry_run(config_path: str) -> int:
    run_started = time.perf_counter()
    stage_start("RUN", "RUN")
    config = load_and_validate_config(config_path)
    log_info("RUN", f"RUN start run_id=auto config={config_path}")
    log_info("CONFIG", f"CONFIG loaded sections={','.join(REQUIRED_SECTIONS)}")
    log_info("CONFIG", "CONFIG validated")
    runs_root = config.raw["compute"]["runs_root"]
    run_context = create_run_context(config_path=config_path, runs_root=runs_root)
    manager = ArtifactManager(runs_root)
    snapshot_path = manager.save_config_snapshot(run_context.run_dir, config_path)
    manifest = {
        "run_id": run_context.run_id,
        "mode": "dry-run-foundation",
        "config_path": str(Path(config_path).resolve()),
        "created_at_utc": run_context.created_at.isoformat(),
        "sections": list(REQUIRED_SECTIONS),
        "run_dir": str(run_context.run_dir),
    }
    manager.save_run_manifest(run_context.run_dir, manifest)
    log_info("ARTIFACTS", f"ARTIFACTS created run_dir={run_context.run_dir}")
    log_info("ARTIFACTS", f"ARTIFACTS snapshot saved path={snapshot_path}")
    log_info("DRYRUN", "DRYRUN done")
    stage_done("RUN", "RUN", elapsed_sec=time.perf_counter() - run_started)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        return run_dry_run(args.config)
    if not args.clickhouse_dsn:
        parser.error("--clickhouse-dsn is required for full-run mode")
    run_pump_end_v2_pipeline(args.config, clickhouse_dsn=args.clickhouse_dsn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
