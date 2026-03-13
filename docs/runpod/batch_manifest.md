# Batch Manifest

Canonical файл: `artifacts/runpod_batches/<batch_id>/batch_manifest.json`

Пример: `scripts/runpod_jobs/batch_manifest.example.json`

## Основные поля

- `batch_id`: уникальный идентификатор batch.
- `created_at`: UTC timestamp.
- `baseline`:
  - `baseline_id`
  - `baseline_hash`
  - `bundle_path`
  - `shared_inputs_hash`
  - `cache_key`
- `shared_inputs`: общие параметры pipeline.
- `pods`: список pod (`alias`, `pod_id`, `ssh_user`).
- `experiments`: список экспериментов.
- `launch_policy`: правила запуска (`one_active_experiment_per_pod`, `tail_only`, `dry_run`).
- `artifact_policy`: какие артефакты скачивать.

## Delta model (experiment.delta)

Поддерживаются:
- `overlay_files`: копировать файл поверх baseline release.
- `changed_files`: alias overlay для маленьких правок.
- `patch_files`: patch-файлы (`patch -p1`).
- `transform_script`: путь к transform script.
- `command_override`: full override команды.
- `params_override`: param-level override.

## Валидация

- `exp_id` уникален.
- `pod_alias` каждого exp должен существовать в `pods`.
- `baseline.baseline_id` обязателен.
