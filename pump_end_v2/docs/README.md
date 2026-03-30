# pump_end_v2 — актуальный архитектурный контракт

Этот документ описывает **то, что реализовано по факту** в текущем `pump_end_v2`, а не исторический план. Он нужен как быстрый onboarding для любого агента или разработчика, который впервые открывает проект и хочет понять:

- как устроена модель целиком;
- где искать узкие места;
- какие артефакты смотреть в первую очередь;
- как запускать новые эксперименты;
- что является частью baseline, а что сознательно **не** входит в baseline.

Документ привязан к текущей кодовой базе, в которой основная orchestration-логика живет в `pump_end_v2/pipeline/run.py`.

---

## 1. Что делает проект

`pump_end_v2` ищет **short-входы в конец пампа**.

Модель работает так:

1. причинно открывает `pump episode`;
2. внутри эпизода каждые 15 минут строит `decision row`;
3. `detector` оценивает, хороший ли это момент для short-входа;
4. `episode-aware policy` выбирает максимум один candidate signal на episode;
5. `gate` решает, пропустить этот сигнал в торговлю или заблокировать;
6. финальный replay исполняет только прошедшие сигналы с `symbol lock`;
7. результат сохраняется в holdout-артефакты и отчеты.

Ключевая идея: модель не учится на A/B-классах и не работает в отдельной event-centric реальности. Она живет в одном **causal stream-based** контракте от train до holdout.

---

## 2. Схема системы сверху вниз

Pipeline по факту выглядит так:

1. **Input loading**
2. **Prepared layers**
3. **Event core**
4. **Detector dataset**
5. **Detector train OOF**
6. **Detector val policy**
7. **Detector test**
8. **Gate val**
9. **Gate test**
10. **Execution replay**
11. **Execution reports**
12. **Artifacts save**

Это именно та последовательность, которую выполняет `run_pump_end_v2_pipeline()`.

Главный файл для понимания end-to-end поведения: **`pump_end_v2/pipeline/run.py`**.

---

## 3. Быстрый map по модулям

Если агент открывает проект впервые, порядок чтения такой:

1. `pump_end_v2/config.py` — конфиг и валидация секций;
2. `pump_end_v2/pipeline/run.py` — orchestration всего run;
3. `pump_end_v2/data/event_opener.py` — causal opener эпизодов;
4. `pump_end_v2/data/decision_rows.py` — decision rows и entry grid;
5. `pump_end_v2/data/resolver.py` — future resolution и target semantics;
6. `pump_end_v2/features/*` — prepared layers и detector feature view;
7. `pump_end_v2/detector/*` — dataset, model, policy, policy search, target metrics;
8. `pump_end_v2/gate/*` — gate feature view, dataset, model, thresholding;
9. `pump_end_v2/execution/replay.py` — replay, TP/SL, symbol lock;
10. `pump_end_v2/execution/metrics.py` — финальные trading reports;
11. `tests/pump_end_v2/*` — текущие системные контракты.

---

## 4. Данные и входы

Проект использует DB-first контракт через секции конфига:

- `data.window` — историческое окно загрузки;
- `data.universe.symbols` — список торгуемых символов;
- `data.references` — референсы BTC/ETH, которые всегда добавляются в загрузку;
- `data.clickhouse` — таблицы `candles` и `transactions`, timezone;
- `--clickhouse-dsn` — runtime-параметр CLI для full-run.

### Что делает каждый слой данных

- **15m** — основной слой для opener, decision rows, detector, gate context и primary execution path;
- **1m** — execution fallback для ambiguous 15m бара, подгружается stage-wise только под нужные candidate windows;
- **1s** — lazy fallback для ambiguous 1m бара, запрашивается точечно из `transactions` по одной минуте.

`1m` и `1s` не предзагружаются в начале run-а.

---

## 5. Prepared layers: что считается заранее

Система строит несколько подготовленных слоев.

### 5.1. `token_state`
Базовый per-symbol, per-15m snapshot. Используется detector-ом и частично gate-ом.

Туда входят, например:

- close returns;
- intrabar range;
- body / wick statistics;
- volatility windows;
- RSI-like / MFI-like / MACD-like признаки;
- dollar volume / liquidity score;
- runup и volume ratio;
- pump context flags.

### 5.2. `reference_state`
Контекст по BTC и ETH:

- returns;
- intrabar range;
- volume ratio;
- pump context flags.

### 5.3. `breadth_state`
Слой ширины рынка:

- universe size;
- advancers share;
- mean / median / std close return;
- near-high share;
- pump-context share;
- volume-spike share.

### 5.4. `episode_state`
Episode-local snapshot, который строится после opener-а:

- `episode_age_bars`;
- `episode_runup_from_open_pct`;
- `episode_extension_from_open_pct`;
- `bars_since_episode_high`;
- `drawdown_from_episode_high_so_far`;
- `high_retest_count`;
- `high_persistence_4`;
- `episode_pump_context_streak`.

Именно `episode_state` связывает event-aware постановку с detector-ом.

---

## 6. Event opener: как появляются episodes

Главный файл: **`pump_end_v2/data/event_opener.py`**.

### Что делает opener

Opener причинно открывает и закрывает `pump episode` по символу.

Он использует:

- `runup_pct`;
- `near_high_flag`;
- `pump_context_flag`;
- `volume_ratio`;
- правила cooldown / expiry / context decay.

### Важные свойства

- opener **не** смотрит вперед;
- opener **не** размечает A/B;
- opener не пытается решить, хороший ли сейчас вход;
- opener задает только episode-frame, внутри которого detector ищет точку входа.

### Ключевые expiry reasons

По коду opener закрывает эпизод по причинам уровня:

- `max_age`;
- `drawdown`;
- `context_decay`;
- `data_end`.

### Что смотреть для качества opener-а

- `prepared/episodes.parquet`
- `prepared/episode_summary.parquet`
- `prepared/event_quality_report.json`

Если проблема уже здесь, detector и gate не спасут run.

---

## 7. Decision rows: что является train-row detector-а

Главный файл: **`pump_end_v2/data/decision_rows.py`**.

### Каноническая train-row

Одна строка detector-а = **один real decision point внутри active episode**.

То есть это не synthetic offset вокруг будущего peak-а, а причинно доступная точка, которую система действительно могла бы увидеть на проде.

### Временная логика

Для каждой row фиксируются:

- `context_bar_open_time`
- `decision_time`
- `entry_bar_open_time`

`entry_bar_open_time` считается через 15m grid и `entry_shift_bars` из execution contract.

Это важный системный контракт: decision-time и entry-time согласованы между train, val, test и execution replay.

---

## 8. Resolver и target semantics

Главный файл: **`pump_end_v2/data/resolver.py`**.

Это один из самых важных модулей проекта.

### Что делает resolver

Для каждой decision row он считает future-resolved семантику сигнала.

Минимальный resolved block:

- `future_prepullback_squeeze_pct`
- `future_pullback_pct`
- `future_net_edge_pct`
- `bars_to_pullback`
- `bars_to_peak_after_row`
- `bars_to_resolution`
- `future_outcome_class`
- `signal_quality_h32`
- `target_good_short_now`
- `target_reason`
- `entry_quality_score`
- `ideal_entry_row_id`
- `ideal_entry_bar_open_time`
- `is_ideal_entry`

### Зафиксированная семантика

- **squeeze** — рост цены после нашей точки входа и до начала pullback;
- **pullback** — откат после squeeze, считаемый относительно точки входа;
- проценты считаются **от entry**, а не от peak.

### `future_outcome_class`

Сейчас это:

- `reversal`
- `continuation`
- `flat`

### `signal_quality_h32`

Resolver сейчас выдает следующие signal-quality классы:

- `clean_retrace_h32`
- `dirty_retrace_h32`
- `clean_no_pullback_h32`
- `dirty_no_pullback_h32`
- `pullback_before_squeeze_h32`

### `target_good_short_now`

В текущей реализации positive row — это row, где итоговый `future_outcome_class == reversal`.

### `target_reason`

Сейчас reason labels строятся как:

- `good`
- `too_early`
- `too_late`
- `continuation`
- `flat`
- `invalid_context`

### Что важно понимать

`ideal_entry_row` в текущем коде используется как **диагностический ориентир**, а не как отдельная модельная задача.

---

## 9. Detector feature view и detector dataset

Главные файлы:

- `pump_end_v2/features/detector_view.py`
- `pump_end_v2/features/manifest.py`
- `pump_end_v2/detector/dataset.py`

### Detector получает только detector-side признаки

Сейчас detector feature space формируется из:

- token snapshot;
- episode snapshot;
- reference state;
- breadth state.

Detector **не** получает:

- strategy state;
- signal flow;
- gate-specific признаки.

### Detector dataset

После join feature view + resolved rows формируется `detector/dataset.parquet`.

Ключевые служебные поля:

- `trainable_row`
- `detector_trainable_row`
- `dataset_split`

### Feature contract

Feature набор detector-а фиксирован кодом через `DETECTOR_FEATURE_COLUMNS`.

В проекте **нет** runtime feature switches. Менять фичи нужно кодом, а не конфигом.

Это сознательное решение: меньше hidden logic, меньше дрейфа экспериментов.

---

## 10. Detector model и OOF training

Главные файлы:

- `pump_end_v2/detector/model.py`
- `pump_end_v2/detector/splits.py`
- `pump_end_v2/detector/policy_search.py`

### Модель

Baseline detector — **CatBoostClassifier**.

### Как detector учится

1. train split используется для fit;
2. на нем же генерируются walk-forward folds;
3. на каждом fold detector обучается только на прошлом;
4. на fold-val строятся **OOF policy rows** и потом **OOF candidate signals**.

Именно эти OOF candidate signals потом питают gate.

### Почему это важно

Gate не учится на in-sample сигнале detector-а. Он получает честный OOF поток.

---

## 11. Detector policy

Главные файлы:

- `pump_end_v2/detector/policy.py`
- `pump_end_v2/detector/policy_search.py`

### Что делает policy

Policy — это **episode-aware state machine**, которая выбирает максимум один signal на episode.

### Важные поля candidate ledger

После policy detector сохраняет поля уровня:

- `signal_id`
- `episode_id`
- `fire_decision_row_id`
- `decision_time`
- `entry_bar_open_time`
- `p_good`
- `peak_p_good_before_fire`
- `p_good_drop_from_peak`
- `episode_age_bars`
- `distance_from_episode_high_pct`
- hindsight diagnostic columns из resolver-а.

### Параметры policy

В текущем baseline detector policy использует ровно три параметра:

- `arm_score_min`
- `fire_score_floor`
- `turn_down_delta`

### Что в baseline отсутствует

- uncertainty gating;
- multi-fire per episode;
- symbol-global state вместо episode-aware state.

### Какие файлы смотреть

- `detector/policy_sweep_val.csv`
- `detector/selected_policy.json`
- `detector/train_oof_candidate_signals.parquet`
- `detector/val_candidate_signals.parquet`
- `detector/test_candidate_signals.parquet`

---

## 12. Detector diagnostics: как понять, слабый ли detector или тупая policy

### Если хочешь понять качество target-learning detector-а
Смотри:

- `detector/val_target_metrics.json`
- `detector/test_target_metrics.json`

Это быстрый ответ на вопрос, как detector разделяет:

- good rows;
- too_early;
- too_late;
- continuation;
- flat.

### Если хочешь понять, тупая ли policy
Смотри:

- `detector/train_oof_policy_metrics.json`
- `detector/val_policy_metrics.json`
- `detector/test_policy_metrics.json`
- `detector/policy_sweep_val.csv`

Особенно важны:

- `good_episode_capture_rate`
- `bad_episode_fire_rate`
- `median_bars_fire_to_ideal`
- `arm_to_fire_conversion`
- `reset_without_fire_share`
- `fires_per_30d`

#### Как читать

- высокий `good_episode_capture_rate` + плохие target metrics → detector weak;
- хорошие target metrics + плохой `good_episode_capture_rate` → policy weak;
- высокий `bad_episode_fire_rate` → policy слишком агрессивная;
- большой `median_bars_fire_to_ideal` → policy опаздывает;
- большой `reset_without_fire_share` → policy слишком часто теряет хорошую зону.

---

## 13. Gate: что это такое по факту

Главные файлы:

- `pump_end_v2/gate/feature_view.py`
- `pump_end_v2/gate/dataset.py`
- `pump_end_v2/gate/model.py`
- `pump_end_v2/gate/pipeline.py`
- `pump_end_v2/gate/threshold.py`

### Роль gate

Gate — это **signal-level keep/block model** поверх detector candidate signals.

Gate не:

- меняет timing сигнала;
- не ищет peak второй раз;
- не pause/resume-машина по всему рынку.

### Что становится train-row gate-а

Одна row = **один detector candidate signal**.

### Feature groups gate-а

Gate feature space фиксирован кодом и состоит из:

1. detector outputs;
2. market context;
3. breadth;
4. signal flow;
5. strategy state.

### Gate model

Baseline gate — **CatBoostClassifier**.

### Gate target

В текущем коде gate использует единственную каноническую binary-цель по
`counterfactual_trade_outcome`:

- `SL` → `target_block_signal = 1` (сигнал надо блокировать);
- `TP` → `target_block_signal = 0` (сигнал надо пропускать);
- `timeout` / `ambiguous` не участвуют в обучении (`gate_trainable_signal = false`).

`signal_quality_h32` после этого трактуется как diagnostic/meta колонка для анализа,
а не как оптимизируемая цель gate.

### `block_reason`

Сейчас gate dataset размечает `block_reason` на основе
`counterfactual_trade_outcome`:

- `block_sl`
- `keep_tp`
- `skip_timeout`
- `skip_ambiguous`
- `skip_unknown`

### Важная реальность текущей реализации

Gate умеет **автоматически отключаться**, если:

- нет trainable signal rows;
- целевой класс вырожден;
- признаки неинформативны;
- score dataset пустой.

В этом случае run не падает, а переходит в fallback `disabled_no_data` режим.

Где это смотреть:

- `reports/run_summary.json`
- `gate/threshold_sweep_val_diagnostic.csv`

---

## 14. Gate thresholding и как понимать, полезен ли gate

Главный файл: **`pump_end_v2/gate/threshold.py`**.

### Policy gate-а

Gate policy сейчас предельно простая:

- считается `p_block`;
- если `p_block >= block_threshold` → сигнал блокируется;
- иначе → keep.

### Что сохраняется

- `gate/threshold_sweep_val_diagnostic.csv`
- `gate/threshold_sweep_val_execution.csv`
- `gate/selected_threshold.json`
- `gate/gate_deciles.csv`
- `gate/gate_deciles_test.csv`

### Что смотреть в первую очередь

#### Если хочешь понять, несет ли gate предиктивную силу
Смотри:

- `gate/gate_deciles.csv`
- `gate/gate_deciles_test.csv`

Если верхние децили `p_block` не выглядят явно хуже нижних — gate не добавляет value.

#### Если хочешь понять, где tradeoff
Смотри:

- `gate/threshold_sweep_val_execution.csv`

Особенно:

- `blocked_by_model_trainable`
- `signals_after_execution`
- `blocked_by_model`
- `blocked_by_symbol_lock`
- `tp_tax_model`
- `sl_capture_model`
- `tp_tax_execution`
- `sl_capture_execution`
- `selection_score`

#### Как читать

- главный критерий полезности gate — model-stage tradeoff:
  выше `sl_capture_model`, ниже `tp_tax_model`, и достаточный `blocked_by_model_trainable`;
- execution-метрики (`pnl_after_execution`, worst windows, streak) используются как tie-breaker между model-stage кандидатами;
- сильное падение `signals_after_execution` без улучшения `pnl_after_execution` — gate слишком агрессивен;
- если gate отключен (`disabled_no_data`), не надо интерпретировать результаты как “gate плохой” — это означает, что текущий detector на этом run не дал пригодной обучающей базы для gate.

---

## 15. Execution layer

Главные файлы:

- `pump_end_v2/execution/replay.py`
- `pump_end_v2/execution/metrics.py`

### Execution contract baseline

Сейчас execution contract задается конфигом:

- `tp_pct`
- `sl_pct`
- `max_hold_bars`
- `entry_shift_bars`

### Replay behavior

- replay идет по `15m` как primary path;
- если в одном 15m баре одновременно задеты TP и SL, выполняется fallback в `1m` внутри этой 15m свечи;
- если в одном 1m баре одновременно задеты TP и SL, выполняется fallback в `1s` через lazy fetch из `transactions`;
- если детализация недоступна, исход фиксируется как `ambiguous`.

### Symbol lock

Финальный replay использует **symbol position lock**:

- если по символу уже есть открытая сделка;
- новый keep-сигнал по этому символу получает `execution_status = blocked_symbol_lock`;
- новая сделка по символу не открывается, пока предыдущая не закрыта.

### Важное различие артефактов

- `candidate_signals.parquet` — все candidate signals после gate decision;
- `test_signals_holdout.csv` — только **реально исполненные** сигналы после gate и symbol lock.

Главный торговый файл — именно второй.

---

## 16. Основные артефакты run-а

Run сохраняет артефакты в структуре:

- `prepared/`
- `detector/`
- `gate/`
- `eval/val/`
- `eval/test/`
- `reports/`

### 16.1. Что смотреть почти всегда

#### Сверху вниз

1. `reports/run_summary.json`
2. `prepared/event_quality_report.json`
3. `detector/val_target_metrics.json`
4. `detector/val_policy_metrics.json`
5. `detector/policy_sweep_val.csv`
6. `gate/threshold_sweep_val_execution.csv`
7. `gate/gate_deciles.csv`
8. `eval/test/metrics_holdout.json`
9. `eval/test/monthly_report.csv`
10. `eval/test/symbol_report.csv`
11. `eval/test/window_report_6h.csv`
12. `eval/test/window_report_24h.csv`

### 16.2. Если нужен raw material для глубокого дебага

- `prepared/episodes.parquet`
- `prepared/decision_rows.parquet`
- `prepared/resolved_rows.parquet`
- `detector/train_oof_policy_rows.parquet`
- `detector/train_oof_candidate_signals.parquet`
- `detector/val_policy_rows.parquet`
- `detector/test_policy_rows.parquet`
- `gate/dataset_train_oof.parquet`
- `gate/dataset_val.parquet`
- `gate/dataset_test.parquet`
- `eval/test/execution_decisions.parquet`

---

## 17. Как искать bottleneck: короткая карта диагностики

### Сценарий 1. Сигналов очень мало уже до gate
Смотри:

- `prepared/event_quality_report.json`
- `detector/policy_sweep_val.csv`
- `detector/val_policy_metrics.json`

Вероятные причины:

- opener открывает мало episodes;
- detector policy слишком зажата;
- detector не находит good zones.

### Сценарий 2. Detector rows выглядят плохо
Смотри:

- `detector/val_target_metrics.json`
- `detector/test_target_metrics.json`

Если `good_row_precision/recall` низкие, а FP по `continuation/flat` высокие — проблема в detector features / model / target separability.

### Сценарий 3. Detector rows хорошие, но signal quality плохая
Смотри:

- `detector/val_target_metrics.json`
- `detector/val_policy_metrics.json`
- `detector/policy_sweep_val.csv`

Если detector rows разделяет неплохо, но `good_episode_capture_rate` низкий или `median_bars_fire_to_ideal` высокий — bottleneck в policy.

### Сценарий 4. Gate ничего не дает
Смотри:

- `reports/run_summary.json` (`gate_status`)
- `gate/gate_deciles.csv`
- `gate/threshold_sweep_val_execution.csv`

Варианты:

- `gate_status = disabled_no_data` — detector не дал пригодной базы;
- deciles плоские — gate feature space не добавляет информации;
- threshold sweep не показывает нормального tradeoff — gate useless на текущем run.

### Сценарий 5. Gate улучшает модельно, но финальный PnL все равно слабый
Смотри:

- `eval/test/metrics_holdout.json`
- `eval/test/monthly_report.csv`
- `eval/test/window_report_6h.csv`
- `eval/test/window_report_24h.csv`
- `eval/test/execution_decisions.parquet`

Проблема может быть в:

- execution contract;
- symbol lock effects;
- недостаточной monthly stability;
- слишком локальных signal improvements без общей торговой ценности.

---

## 18. Как прогонять новые эксперименты

### Принцип

Все новые эксперименты должны запускаться **только через отдельные TOML-конфиги**.

Базовый шаблон можно держать в `pump_end_v2/config/base.toml`, но реальные эксперименты должны лежать отдельно, например:

- `pump_end_v2/config/mini_baseline_v1.toml`
- `pump_end_v2/config/exp_001_threshold_density.toml`
- `pump_end_v2/config/exp_002_gate_conservative.toml`

### Что должен делать агент

1. **Не редактировать `base.toml` под каждый запуск.**
2. Создавать новый конфиг в `pump_end_v2/config/`.
3. Менять только:
   - `data.window` (историческое окно);
   - `data.universe.symbols` (набор торгуемых символов);
   - `data.clickhouse` (имена таблиц и timezone);
   - splits;
   - event opener параметры;
   - resolver параметры;
   - detector model / cv / policy параметры;
   - gate threshold / model параметры;
   - execution contract;
   - runs_root.
4. **Не пытаться управлять feature set через конфиг.** Этого в текущей архитектуре нет.
5. После run сначала читать:
   - `reports/run_summary.json`
   - `prepared/event_quality_report.json`
   - `detector/policy_sweep_val.csv`
   - `gate/threshold_sweep_val_execution.csv`
   - `eval/test/metrics_holdout.json`
6. Только потом лезть в parquet-артефакты.

### Рекомендуемый порядок изменений по run-ам

#### 1. Сначала стабилизировать detector
Пока detector не дает осмысленный candidate stream, gate трогать бессмысленно.

#### 2. Потом трогать gate
Если detector стал приличным, уже смотреть на gate threshold и gate model.

#### 3. Execution contract менять редко
Если baseline execution contract меняется слишком часто, сравнивать run-ы становится тяжело.

---

## 19. Базовые команды

### Dry-run только на валидацию конфига и run-dir

```bash
uv run python -m pump_end_v2.cli.run_pump_end_v2 --config pump_end_v2/config/<name>.toml --dry-run
```

### Полный run

```bash
uv run python -m pump_end_v2.cli.run_pump_end_v2 --config pump_end_v2/config/<name>.toml --clickhouse-dsn "<dsn>"
```

---

## 20. Что важно помнить агенту перед любыми изменениями

1. **Главный торговый артефакт — `eval/test/test_signals_holdout.csv`.**
2. `candidate_signals.parquet` и `test_signals_holdout.csv` — это разные сущности.
3. `gate_status = disabled_no_data` — это не баг, а честный fallback режима без trainable gate dataset.
4. Detector и gate intentionally обучаются **каскадом**, а не joint black-box моделью.
5. Feature composition фиксируется кодом, а не config flags.
6. Если run сломался до старта модели, сначала проверять доступность ClickHouse (DSN/сеть/таблицы/таймзону), а не detector/gate логику.
7. Любое изменение надо оценивать не по одной метрике, а минимум по связке:
   - event quality;
   - detector target metrics;
   - detector policy metrics;
   - gate threshold sweep / deciles;
   - final holdout execution metrics.

---

## 21. Краткая суть проекта в одном абзаце

`pump_end_v2` — это причинно-честный event-aware pipeline для short-входов в конец пампа: opener открывает pump episodes, detector внутри них ищет момент входа, episode-aware policy выбирает максимум один candidate signal на episode, gate фильтрует этот сигнал по рынку и состоянию потока, а финальная торговая оценка всегда делается на одном stream-based holdout с фиксированным execution contract и symbol lock.
