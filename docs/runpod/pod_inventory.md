# Pod Inventory

Файл: `pod_inventory.json` (пример в `scripts/runpod_jobs/pod_inventory.example.json`)

```json
{
  "pods": [
    {"alias": "pod1", "pod_id": "abc", "ssh_user": "root"},
    {"alias": "pod2", "pod_id": "def", "ssh_user": "root"}
  ]
}
```

## Правила

- `alias` должен совпадать с `experiments[].pod_alias`.
- `pod_id` должен быть активным RunPod pod.
- `ssh_user` по умолчанию `root`.

## Сценарий "уже есть 5 pod"

1. Заполнить `pod_inventory.json` из текущих pod id.
2. Указать те же alias в `experiment_specs`.
3. Выполнить `launch_batch` с `--pod-inventory`.
