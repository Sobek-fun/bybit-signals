#!/usr/bin/env bash
set -euo pipefail

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[run_experiment][$(ts)] $*"; }

SRC_DIR=""
RUN_DIR=""
VENV_DIR=""
LOG_PATH=""
STARTED_AT_PATH=""
FINISHED_AT_PATH=""
EXIT_CODE_PATH=""
LAUNCH_COMMAND_PATH=""
PYTHON_BIN="python3"
REQUIREMENTS_FILE="scripts/runpod_jobs/requirements_runpod.txt"
CLICKHOUSE_DSN_ENV=""
DETECTOR_DIR_REMOTE=""
TOKENS_FILE_REMOTE=""
SETUP_COMMAND=""
LAUNCH_COMMAND=""
LOCK_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir) SRC_DIR="$2"; shift 2;;
    --run-dir) RUN_DIR="$2"; shift 2;;
    --venv-dir) VENV_DIR="$2"; shift 2;;
    --log-path) LOG_PATH="$2"; shift 2;;
    --started-at-path) STARTED_AT_PATH="$2"; shift 2;;
    --finished-at-path) FINISHED_AT_PATH="$2"; shift 2;;
    --exit-code-path) EXIT_CODE_PATH="$2"; shift 2;;
    --launch-command-path) LAUNCH_COMMAND_PATH="$2"; shift 2;;
    --python-bin) PYTHON_BIN="$2"; shift 2;;
    --requirements-file) REQUIREMENTS_FILE="$2"; shift 2;;
    --clickhouse-dsn-env) CLICKHOUSE_DSN_ENV="$2"; shift 2;;
    --detector-dir-remote) DETECTOR_DIR_REMOTE="$2"; shift 2;;
    --tokens-file-remote) TOKENS_FILE_REMOTE="$2"; shift 2;;
    --setup-command) SETUP_COMMAND="$2"; shift 2;;
    --launch-command) LAUNCH_COMMAND="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "$SRC_DIR" || -z "$RUN_DIR" || -z "$VENV_DIR" || -z "$LOG_PATH" || -z "$STARTED_AT_PATH" || -z "$FINISHED_AT_PATH" || -z "$EXIT_CODE_PATH" || -z "$LAUNCH_COMMAND_PATH" || -z "$LAUNCH_COMMAND" ]]; then
  log "required args are missing"
  exit 2
fi

log "init src_dir=$SRC_DIR run_dir=$RUN_DIR venv_dir=$VENV_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STARTED_AT_PATH"
log "started_at written to $STARTED_AT_PATH"

on_exit() {
  local code=$?
  if [[ -n "$LOCK_DIR" ]]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$FINISHED_AT_PATH"
  echo "$code" > "$EXIT_CODE_PATH"
}
trap on_exit EXIT

if [[ ! -f "$SRC_DIR/$REQUIREMENTS_FILE" ]]; then
  log "requirements file not found: $SRC_DIR/$REQUIREMENTS_FILE"
  exit 3
fi
if [[ -n "$DETECTOR_DIR_REMOTE" && ! -d "$DETECTOR_DIR_REMOTE" ]]; then
  log "missing detector dir: $DETECTOR_DIR_REMOTE"
  exit 4
fi
if [[ -n "$TOKENS_FILE_REMOTE" && ! -f "$TOKENS_FILE_REMOTE" ]]; then
  log "missing tokens file: $TOKENS_FILE_REMOTE"
  exit 5
fi
if [[ -n "$CLICKHOUSE_DSN_ENV" && -z "${!CLICKHOUSE_DSN_ENV:-}" ]]; then
  log "missing required env var: $CLICKHOUSE_DSN_ENV"
  exit 6
fi

cd "$SRC_DIR"
log "switched to src dir"
LOCK_DIR="${VENV_DIR}.lock"
log "acquiring venv lock: $LOCK_DIR"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 1; done
if [[ -x "$VENV_DIR/bin/python" && -f "$VENV_DIR/.ready" ]]; then
  log "shared venv already ready, skip install"
else
  log "shared venv missing/incomplete, provisioning"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
  fi
  python -m pip install -r "$SRC_DIR/$REQUIREMENTS_FILE"
  touch "$VENV_DIR/.ready"
  log "shared venv provisioning done"
fi
source "$VENV_DIR/bin/activate"
rmdir "$LOCK_DIR" 2>/dev/null || true
LOCK_DIR=""
log "venv activated"
if [[ -n "$SETUP_COMMAND" ]]; then
  log "running setup command"
  bash -lc "$SETUP_COMMAND"
fi
export PYTHONPATH="$SRC_DIR"
export PYTHONUNBUFFERED=1
echo "$LAUNCH_COMMAND" > "$LAUNCH_COMMAND_PATH"
log "launch command saved to $LAUNCH_COMMAND_PATH"
log "starting pipeline"
bash -lc "$LAUNCH_COMMAND"
