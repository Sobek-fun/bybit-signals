# RunPod Minimal Flow

Только три команды: `doctor`, `launch`, `relaunch`.

## Формат minimal spec

Файлы примеров:

- regime: `scripts/runpod_jobs/experiment_specs.example.json`
- threshold: `scripts/runpod_jobs/experiment_specs.threshold.example.json`

- `batch_id`: идентификатор батча.
- `runtime.workspace_root`: обычно `/workspace/experiments`.
- `runtime.requirements_file`: путь внутри upload-snapshot.
- `runtime.pipeline_command`: команда пайплайна (всегда использует `RUN_ROOT`, опционально `DETECTOR_DIR`, `TOKENS_FILE`).
- `runtime.clickhouse_dsn_env`: optional имя env-переменной с DSN (например `CH_DB`).
- `runtime.detector_dir_remote`: optional путь к detector на pod.
- `runtime.tokens_file_remote`: optional путь к tokens-файлу на pod.
- `runtime.extra_env`: optional словарь env-переменных, экспортируется перед запуском пайплайна.
- `experiments[]`: `exp_id`, `pod_alias`, `patch_files[]` (максимум один patch на `exp_id`).

Важно:
- multi-patch для одного эксперимента запрещен — только один patch-файл или пустой список.
- launcher всегда использует единый shared venv: `/workspace/.venvs/bybit-signals-runpod`.
- перед deploy выполняется bootstrap shared venv на одном из pod, после чего все эксперименты используют это же окружение без повторной установки зависимостей.

### Regime job

- Обычно нужны `detector_dir_remote`, `tokens_file_remote`, `clickhouse_dsn_env`.
- В `pipeline_command` используется экспорт сигналов и `pump_end_threshold.cli.train_regime_guard`.

### Threshold job

- Достаточно готового parquet, detector/tokens не обязательны.
- Рекомендуется передавать путь через `runtime.extra_env.DATASET_PARQUET`.
- Для запуска threshold используй `python -m pump_end_threshold.cli.train_pump_end_model`, а не `src.scripts...`.
- `clickhouse_dsn_env` заполняй только если нужны trade-replay метрики.

## Формат pod inventory

Файл: `scripts/runpod_jobs/pod_inventory.example.json`

Для каждого alias:
- либо `host` + `port` (прямой SSH),
- либо `pod_id` (SSH endpoint будет резолвиться через RunPod API).

## Команда launch

```bash
python -m scripts.runpod_jobs.cli launch \
  --spec-file scripts/runpod_jobs/experiment_specs.threshold.example.json \
  --pod-inventory scripts/runpod_jobs/pod_inventory.example.json \
  --runpod-api-key "$RUNPOD_API_KEY" \
  --ssh-key-path ~/.ssh/id_ed25519
```

В stdout печатаются только уникальные `tail -f` команды для каждого эксперимента.
Дополнительно launcher печатает этапные логи (`assemble`, `venv`, `deploy`, `launch`) для быстрой диагностики.

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
