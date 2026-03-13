#!/usr/bin/env bash
set -euo pipefail

EXP_ID=""
RUN_DIR=""
RELEASE_DIR=""
VENV_DIR=""
TMP_DIR=""
BASELINE_HASH=""
CACHE_ROOT="/workspace/experiments/_baseline_cache"
LOCKS_ROOT="/workspace/experiments/_locks"
CLICKHOUSE_DSN=""
DETECTOR_DIR="artifacts/tune_threshold_no_argmax_liq7d_detector"
TOKENS_FILE="config/regime_tokens_curated55.txt"
TRANSFORM_SCRIPT="scripts/runpod_jobs/transform_template.py"
TARGET_COL=""
TRAIN_END=""
OOS_END=""
START_DATE=""
BUILD_DATASET_ARGS_JSON="{}"
TRAIN_ARGS_JSON="{}"
PARAMS_OVERRIDE_JSON="{}"
COMMAND_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id) EXP_ID="$2"; shift 2;;
    --run-dir) RUN_DIR="$2"; shift 2;;
    --release-dir) RELEASE_DIR="$2"; shift 2;;
    --venv-dir) VENV_DIR="$2"; shift 2;;
    --tmp-dir) TMP_DIR="$2"; shift 2;;
    --baseline-hash) BASELINE_HASH="$2"; shift 2;;
    --cache-root) CACHE_ROOT="$2"; shift 2;;
    --locks-root) LOCKS_ROOT="$2"; shift 2;;
    --clickhouse-dsn) CLICKHOUSE_DSN="$2"; shift 2;;
    --detector-dir) DETECTOR_DIR="$2"; shift 2;;
    --tokens-file) TOKENS_FILE="$2"; shift 2;;
    --transform-script) TRANSFORM_SCRIPT="$2"; shift 2;;
    --target-col) TARGET_COL="$2"; shift 2;;
    --train-end) TRAIN_END="$2"; shift 2;;
    --oos-end) OOS_END="$2"; shift 2;;
    --start-date) START_DATE="$2"; shift 2;;
    --build-dataset-args-json) BUILD_DATASET_ARGS_JSON="$2"; shift 2;;
    --train-args-json) TRAIN_ARGS_JSON="$2"; shift 2;;
    --params-override-json) PARAMS_OVERRIDE_JSON="$2"; shift 2;;
    --command-override) COMMAND_OVERRIDE="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "$EXP_ID" || -z "$RUN_DIR" || -z "$RELEASE_DIR" || -z "$VENV_DIR" || -z "$TMP_DIR" || -z "$BASELINE_HASH" ]]; then
  echo "required args are missing"
  exit 2
fi
if [[ -z "$CLICKHOUSE_DSN" ]]; then
  echo "clickhouse dsn is required"
  exit 2
fi

mkdir -p "$RUN_DIR" "$TMP_DIR" "$LOCKS_ROOT"
rm -rf "$VENV_DIR"

ensure_uv() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on pod"
    exit 3
  fi
  if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --upgrade pip >/tmp/runpod_uv_bootstrap.log 2>&1
    python3 -m pip install uv >>/tmp/runpod_uv_bootstrap.log 2>&1
  fi
}

create_cache_if_needed() {
  local cache_dir="$CACHE_ROOT/$BASELINE_HASH"
  local lock_dir="$LOCKS_ROOT/baseline_${BASELINE_HASH}.lock"
  local marker="$cache_dir/done.marker"
  local manifest="$cache_dir/cache_manifest.json"
  local raw="$cache_dir/raw_detector_signals_curated55.parquet"
  local base="$cache_dir/regime_dataset_base.parquet"

  mkdir -p "$CACHE_ROOT" "$LOCKS_ROOT"
  if [[ -f "$marker" && -f "$manifest" && -f "$raw" && -f "$base" ]]; then
    return 0
  fi

  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 3
    if [[ -f "$marker" && -f "$manifest" && -f "$raw" && -f "$base" ]]; then
      return 0
    fi
  done

  trap 'rmdir "$lock_dir" >/dev/null 2>&1 || true' EXIT
  mkdir -p "$cache_dir"

  local work="$TMP_DIR/baseline_cache_build"
  rm -rf "$work"
  mkdir -p "$work"

  local det="$RELEASE_DIR/$DETECTOR_DIR"
  local tok="$RELEASE_DIR/$TOKENS_FILE"

  uv venv --python 3.13 "$VENV_DIR" >/dev/null
  export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
  export UV_LINK_MODE=copy

  uv run --python 3.13 python -m pump_end_threshold.cli.export_pump_end_signals \
    --start-date "${START_DATE:-2025-01-01 00:00:00}" \
    --end-date "${OOS_END:-2026-03-03 23:59:59}" \
    --clickhouse-dsn "$CLICKHOUSE_DSN" \
    --model-dir "$det" \
    --symbols-file "$tok" \
    --run-dir "$work" \
    --output "$work/_detector_signals_curated55.csv" \
    --raw-signals-output "$work/raw_detector_signals_curated55.parquet" \
    --skip-guard \
    --workers 8

  uv run --python 3.13 python -m pump_end_threshold.cli.build_regime_dataset \
    --clickhouse-dsn "$CLICKHOUSE_DSN" \
    --signals-path "$work/raw_detector_signals_curated55.parquet" \
    --symbols-file "$tok" \
    --fixed-universe-file "$tok" \
    --run-dir "$work" \
    --output "$work/regime_dataset_base.parquet" \
    --top-n-universe 55 \
    --tp-pct 4.5 \
    --sl-pct 10.0 \
    --max-horizon-bars 200 \
    --trade-replay-source 1s \
    --target-col target_pause_value_next_12h \
    --target-profile pause_value_12h_v2_curated \
    --target-min-resolved 3 \
    --target-sl-rate-threshold 0.55 \
    --feature-profile regime_compact_v4

  cp -f "$work/raw_detector_signals_curated55.parquet" "$raw"
  cp -f "$work/regime_dataset_base.parquet" "$base"
  cat >"$manifest" <<EOF
{
  "baseline_hash": "$BASELINE_HASH",
  "files": [
    "raw_detector_signals_curated55.parquet",
    "regime_dataset_base.parquet"
  ]
}
EOF
  echo "ok" > "$marker"
  rmdir "$lock_dir" >/dev/null 2>&1 || true
  trap - EXIT
}

materialize_run_inputs() {
  local cache_dir="$CACHE_ROOT/$BASELINE_HASH"
  cp -f "$cache_dir/raw_detector_signals_curated55.parquet" "$RUN_DIR/raw_detector_signals_curated55.parquet"
  cp -f "$cache_dir/regime_dataset_base.parquet" "$RUN_DIR/regime_dataset_base.parquet"
}

run_pipeline() {
  local tok="$RELEASE_DIR/$TOKENS_FILE"
  local transform="$RELEASE_DIR/$TRANSFORM_SCRIPT"

  uv run --python 3.13 python "$transform" --exp-id "$EXP_ID" --run-root "$RUN_DIR"

  local target="$TARGET_COL"
  if [[ -z "$target" && -f "$RUN_DIR/target_col.txt" ]]; then
    target="$(tr -d '\r' < "$RUN_DIR/target_col.txt")"
  fi
  if [[ -z "$target" ]]; then
    target="target_pause_value_next_12h"
  fi

  if [[ -n "$COMMAND_OVERRIDE" ]]; then
    bash -lc "$COMMAND_OVERRIDE"
    return 0
  fi

  uv run --python 3.13 python -m pump_end_threshold.cli.train_regime_guard \
    --run-dir "$RUN_DIR" \
    --dataset-parquet "$RUN_DIR/regime_dataset_train.parquet" \
    --target-col "$target" \
    --train-end "${TRAIN_END:-2026-02-20}" \
    --time-budget-min 60 \
    --fold-days 21 \
    --min-train-days 120 \
    --embargo-hours 12 \
    --iterations 3500 \
    --early-stopping-rounds 250 \
    --seed 42 \
    --score-mode block_value \
    --max-blocked-share 0.18 \
    --min-signal-keep-rate 0.80 \
    --min-valid-folds 4 \
    --policy-grid aggressive \
    --model-selection-mode downstream_cv \
    --disable-auto-class-weights

  uv run --python 3.13 python "$RELEASE_DIR/scripts/regime_run_report.py" "$RUN_DIR" --output "$RUN_DIR/run_report.md"

  cat >"$RUN_DIR/artifacts_manifest.json" <<EOF
{
  "exp_id": "$EXP_ID",
  "ready": true,
  "files": [
    "run_report.md",
    "pipeline.log",
    "run_state.json",
    "summary.json"
  ]
}
EOF
}

ensure_uv
create_cache_if_needed
materialize_run_inputs
run_pipeline
