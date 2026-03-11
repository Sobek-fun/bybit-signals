# RunPod Regime Launcher

## What is unified

- `scripts/runpod_regime_batch.py` launches any number of experiments from a JSON spec.
- `scripts/runpod_regime_4pods.py` is a compatibility wrapper with the previous default 4 experiments.
- `scripts/runpod_jobs/run_experiment.sh` is a single remote pipeline runner.
- `scripts/runpod_jobs/apply_experiment_transform.py` contains experiment-specific dataset/target transforms.
- `scripts/runpod_jobs/transform_template.py` is a minimal template for new custom transforms.

## Spec format

Use a JSON array:

```json
[
  {
    "exp_id": "exp1",
    "run_root": "/workspace/experiments/exp1_curated55_blockvalue_strict_notpsl"
  },
  {
    "exp_id": "exp_custom",
    "run_root": "/workspace/experiments/exp_custom",
    "transform_script": "scripts/runpod_jobs/apply_experiment_transform.py",
    "target_col": "target_pause_value_next_12h"
  }
]
```

Required:
- `exp_id`
- `run_root`

Optional:
- `transform_script`
- `target_col`

`transform_script` is passed to `run_experiment.sh` and must produce `regime_dataset_train.parquet`.

If `target_col` is not provided in spec, `run_experiment.sh` reads it from `<run_root>/target_col.txt`.

## Default transform contract

The default script `apply_experiment_transform.py`:

- reads `<run_root>/regime_dataset_base.parquet`
- writes `<run_root>/regime_dataset_train.parquet`
- writes `<run_root>/target_col.txt`
- prints target column

## Launch examples

Generic launch:

```bash
python scripts/runpod_regime_batch.py \
  --runpod-api-key "<RUNPOD_API_KEY>" \
  --spec-file scripts/runpod_jobs/experiment_specs.example.json \
  --storage-id e4sm7sqxod \
  --clickhouse-dsn "http://admin:PASS@185.189.45.79:8123/bybit"
```

Legacy 4-exp launch:

```bash
python scripts/runpod_regime_4pods.py \
  --runpod-api-key "<RUNPOD_API_KEY>" \
  --experiments exp1,exp2,exp3,exp4
```

## Monitoring and download

- Each started experiment prints `tail_command`.
- Result summary is saved to `artifacts/runpod_exports/launched_batch_<timestamp>.json`.
- Download artifacts with `scp -r root@<host>:<run_root> artifacts/`.
