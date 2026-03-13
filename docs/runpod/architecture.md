# Architecture

## Invariants

1. Baseline - отдельная first-class сущность.
2. Experiment = immutable baseline + delta.
3. Нет shared mutable source dir между параллельными экспериментами.
4. Venv всегда в pod-local `/tmp`, не в network volume.
5. Status не строится на PID оболочки `nohup`.
6. Нет `.replace(...)` surgery для shell-команд как основного механизма.
7. Shared cache только через `hash + lock + manifest + done.marker`.
8. Никаких `rm -rf` shared cache path в runtime.
9. По умолчанию: 1 pod = 1 active experiment.

## Path model

### Local control-plane

`artifacts/runpod_batches/<batch_id>/`
- `batch_manifest.json`
- `launch_results.json`
- `batch_status.json`
- `downloaded/`
- `logs/`

### Pod-local ephemeral

`/tmp/bybit-signals/<batch_id>/<exp_id>/`
- `release/` (code snapshot)
- `venv/`
- `tmp/`

### Persistent volume paths

`/workspace/experiments/<batch_id>/<exp_id>/`
- `run_state.json`
- `launch_manifest.json`
- `pid.txt`
- `exit_code.txt`
- `started_at.txt`
- `finished_at.txt`
- `pipeline.log`
- `artifacts_manifest.json`

Shared cache:
- `/workspace/experiments/_baseline_cache/<baseline_hash>/`
- `/workspace/experiments/_locks/`

## State machine

- `PREPARING`
- `DEPLOYING`
- `BOOTSTRAPPING`
- `RUNNING`
- `FINISHED`
- `FAILED`
- `UNKNOWN`

Пишется в `run_state.json` через `remote_launcher.py`, heartbeat обновляется в процессе исполнения.

## Pipeline phases

1. `prepare_batch`: baseline bundle + baseline hash + canonical manifest.
2. `launch_batch`: isolated release deployment + delta apply + detached launch.
3. `status_batch`: read remote `run_state.json` + artifacts readiness.
4. `download_batch`: pull canonical artifacts from launch metadata.
5. `relaunch_experiment`: relaunch single exp без влияния на остальные.
