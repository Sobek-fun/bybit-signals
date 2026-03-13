# Operator Checklist

## Preflight

- [ ] `RUNPOD_API_KEY` задан в env.
- [ ] `CLICKHOUSE_DSN` задан в env/spec.
- [ ] SSH key существует и доступен.
- [ ] `pod_inventory.json` заполнен.
- [ ] `doctor` проходит без критических ошибок.

## Launch

- [ ] Выполнен `prepare_batch`.
- [ ] Проверен `dry_run_plan.json`.
- [ ] Запуск через `launch_batch --tail-only`.
- [ ] Сохранен `launch_results.json`.

## Runtime

- [ ] Проверка `status_batch` отдельной командой.
- [ ] Без авто-download во время launch.
- [ ] Для падения одного exp использовать только `relaunch_experiment`.

## Post-run

- [ ] `download_batch` сохранил артефакты в `downloaded/`.
- [ ] Проверены `run_report.md`, `pipeline.log`, `run_state.json`.
- [ ] `batch_status.json` архивирован вместе с manifest.
