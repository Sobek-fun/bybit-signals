# RunPod Minimal Flow

Только три команды: `doctor`, `launch`, `relaunch`.

## Формат minimal spec

Файл: `scripts/runpod_jobs/experiment_specs.example.json`

- `batch_id`: идентификатор батча.
- `runtime.workspace_root`: обычно `/workspace/experiments`.
- `runtime.requirements_file`: путь внутри upload-snapshot.
- `runtime.pipeline_command`: базовая команда пайплайна (использует `RUN_ROOT`, `DETECTOR_DIR`, `TOKENS_FILE`).
- `runtime.clickhouse_dsn_env`: имя env-переменной на pod (например `CH_DB`).
- `runtime.detector_dir_remote`: внешний путь на pod к detector.
- `runtime.tokens_file_remote`: внешний путь на pod к tokens-файлу.
- `experiments[]`: `exp_id`, `pod_alias`, `patch_files[]`.

## Формат pod inventory

Файл: `scripts/runpod_jobs/pod_inventory.example.json`

Для каждого alias:
- либо `host` + `port` (прямой SSH),
- либо `pod_id` (SSH endpoint будет резолвиться через RunPod API).

## Команда launch

```bash
python -m scripts.runpod_jobs.cli launch \
  --spec-file scripts/runpod_jobs/experiment_specs.example.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key "$RUNPOD_API_KEY" \
  --ssh-key-path ~/.ssh/id_ed25519
```

В stdout печатаются только уникальные `tail -f` команды для каждого эксперимента.

## Команда relaunch (один experiment)

```bash
python -m scripts.runpod_jobs.cli relaunch \
  --spec-file scripts/runpod_jobs/experiment_specs.example.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key "$RUNPOD_API_KEY" \
  --ssh-key-path ~/.ssh/id_ed25519 \
  --exp-id patch_a
```

`relaunch` удаляет только workspace одного эксперимента, пересобирает и перезапускает только его.

## Что делает пользователь вручную

- Следит за логом через выведенную `tail -f` команду.
- Сам скачивает нужные артефакты.
- Сам останавливает pod после завершения.
