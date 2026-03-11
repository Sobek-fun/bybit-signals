#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?exp id required}"
RUN_ROOT="${2:?run root required}"
REPO_DIR="${3:?repo dir required}"
CH_DB="${4:?clickhouse dsn required}"
DETECTOR_DIR="${5:?detector dir required}"
TOKENS_FILE="${6:?tokens file required}"
TRANSFORM_SCRIPT="${7:-scripts/runpod_jobs/apply_experiment_transform.py}"
TARGET_COL_OVERRIDE="${8:-}"
WORKERS="${WORKERS:-8}"
TRAIN_END="2026-02-20"
OOS_END="2026-03-03 23:59:59"
SHARED_DIR="/workspace/experiments/shared"
SHARED_RAW="$SHARED_DIR/raw_detector_signals_curated55.parquet"
SHARED_CSV="$SHARED_DIR/_detector_signals_curated55.csv"

export RUN_ROOT

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade pip >/tmp/regime_bootstrap.log 2>&1
  python -m pip install uv >>/tmp/regime_bootstrap.log 2>&1
fi

mkdir -p "$SHARED_DIR"

if [ "$EXP_ID" = "exp1" ] || [ ! -f "$SHARED_RAW" ]; then
  set +e
  uv run --python 3.13 python -m pump_end_threshold.cli.export_pump_end_signals \
    --start-date "2025-01-01 00:00:00" \
    --end-date "$OOS_END" \
    --clickhouse-dsn "$CH_DB" \
    --model-dir "$DETECTOR_DIR" \
    --symbols-file "$TOKENS_FILE" \
    --run-dir "$RUN_ROOT" \
    --output "$RUN_ROOT/_detector_signals_curated55.csv" \
    --raw-signals-output "$RUN_ROOT/raw_detector_signals_curated55.parquet" \
    --skip-guard \
    --workers "$WORKERS"
  export_rc=$?
  set -e
  if [ ! -f "$RUN_ROOT/raw_detector_signals_curated55.parquet" ]; then
    echo "export failed and raw parquet missing, rc=$export_rc"
    exit 1
  fi
  cp -f "$RUN_ROOT/raw_detector_signals_curated55.parquet" "$SHARED_RAW"
  if [ -f "$RUN_ROOT/_detector_signals_curated55.csv" ]; then
    cp -f "$RUN_ROOT/_detector_signals_curated55.csv" "$SHARED_CSV"
  fi
else
  cp -f "$SHARED_RAW" "$RUN_ROOT/raw_detector_signals_curated55.parquet"
  if [ -f "$SHARED_CSV" ]; then
    cp -f "$SHARED_CSV" "$RUN_ROOT/_detector_signals_curated55.csv"
  fi
fi

cd "$REPO_DIR"

uv run --python 3.13 python -m pump_end_threshold.cli.build_regime_dataset \
  --clickhouse-dsn "$CH_DB" \
  --signals-path "$RUN_ROOT/raw_detector_signals_curated55.parquet" \
  --symbols-file "$TOKENS_FILE" \
  --fixed-universe-file "$TOKENS_FILE" \
  --run-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/regime_dataset_base.parquet" \
  --top-n-universe 55 \
  --tp-pct 4.5 \
  --sl-pct 10.0 \
  --max-horizon-bars 200 \
  --trade-replay-source "1s" \
  --target-col "target_pause_value_next_12h" \
  --target-profile "pause_value_12h_v2_curated" \
  --target-min-resolved 3 \
  --target-sl-rate-threshold 0.55 \
  --feature-profile "regime_compact_v4"

uv run --python 3.13 python "$TRANSFORM_SCRIPT" \
  --exp-id "$EXP_ID" \
  --run-root "$RUN_ROOT"

TARGET_COL="$TARGET_COL_OVERRIDE"
if [ -z "$TARGET_COL" ]; then
  TARGET_COL="$(cat "$RUN_ROOT/target_col.txt")"
fi

uv run --python 3.13 python -m pump_end_threshold.cli.train_regime_guard \
  --run-dir "$RUN_ROOT" \
  --dataset-parquet "$RUN_ROOT/regime_dataset_train.parquet" \
  --target-col "$TARGET_COL" \
  --train-end "$TRAIN_END" \
  --time-budget-min 60 \
  --fold-days 21 \
  --min-train-days 120 \
  --embargo-hours 12 \
  --iterations 3500 \
  --early-stopping-rounds 250 \
  --seed 42 \
  --score-mode "block_value" \
  --max-blocked-share 0.18 \
  --min-signal-keep-rate 0.80 \
  --min-valid-folds 4 \
  --policy-grid "aggressive" \
  --disable-auto-class-weights

uv run --python 3.13 python scripts/regime_run_report.py "$RUN_ROOT" --output "$RUN_ROOT/run_report.md"
