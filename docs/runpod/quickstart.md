# Quickstart

## 1) Подготовить baseline и batch manifest

```bash
python -m scripts.runpod_jobs.cli prepare_batch \
  --spec-file scripts/runpod_jobs/experiment_specs.example.json
```

Результат:
- `artifacts/runpod_batches/<batch_id>/batch_manifest.json`
- `artifacts/runpod_batches/<batch_id>/baseline/baseline_manifest.json`
- `artifacts/runpod_batches/<batch_id>/dry_run_plan.json`

## 2) Запустить 5 экспериментов на 5 существующих pod

```bash
python -m scripts.runpod_jobs.cli launch_batch \
  --batch-manifest artifacts/runpod_batches/<batch_id>/batch_manifest.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key $RUNPOD_API_KEY \
  --ssh-key-path ~/.ssh/id_ed25519 \
  --tail-only
```

В `--tail-only` mode stdout печатает только по одной `tail -f` команде на experiment.

Опционально можно разрешить создание недостающих pod:

```bash
python -m scripts.runpod_jobs.cli launch_batch \
  --batch-manifest artifacts/runpod_batches/<batch_id>/batch_manifest.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key $RUNPOD_API_KEY \
  --pod-template-file scripts/runpod_jobs/examples/pod_create_template.json \
  --create-missing-pods
```

## 3) Проверить статус

```bash
python -m scripts.runpod_jobs.cli status_batch \
  --batch-manifest artifacts/runpod_batches/<batch_id>/batch_manifest.json \
  --runpod-api-key $RUNPOD_API_KEY \
  --ssh-key-path ~/.ssh/id_ed25519
```

Сводка пишется в:
- `artifacts/runpod_batches/<batch_id>/batch_status.json`

## 4) Скачать результаты

```bash
python -m scripts.runpod_jobs.cli download_batch \
  --batch-manifest artifacts/runpod_batches/<batch_id>/batch_manifest.json \
  --runpod-api-key $RUNPOD_API_KEY \
  --ssh-key-path ~/.ssh/id_ed25519
```

Артефакты:
- `artifacts/runpod_batches/<batch_id>/downloaded/<exp_id>/...`

## 5) Перезапустить только один failed experiment

```bash
python -m scripts.runpod_jobs.cli relaunch_experiment \
  --batch-manifest artifacts/runpod_batches/<batch_id>/batch_manifest.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key $RUNPOD_API_KEY \
  --ssh-key-path ~/.ssh/id_ed25519 \
  --exp-id exp3 \
  --tail-only
```
