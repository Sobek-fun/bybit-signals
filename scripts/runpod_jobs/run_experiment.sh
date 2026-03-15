#!/usr/bin/env bash
set -euo pipefail

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
CLICKHOUSE_DSN_ENV="CH_DB"
DETECTOR_DIR_REMOTE=""
TOKENS_FILE_REMOTE=""
SETUP_COMMAND=""
LAUNCH_COMMAND=""

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

if [[ -z "$SRC_DIR" || -z "$RUN_DIR" || -z "$VENV_DIR" || -z "$LOG_PATH" || -z "$STARTED_AT_PATH" || -z "$FINISHED_AT_PATH" || -z "$EXIT_CODE_PATH" || -z "$LAUNCH_COMMAND_PATH" || -z "$DETECTOR_DIR_REMOTE" || -z "$TOKENS_FILE_REMOTE" || -z "$LAUNCH_COMMAND" ]]; then
  echo "required args are missing"
  exit 2
fi

mkdir -p "$RUN_DIR"
rm -rf "$VENV_DIR"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STARTED_AT_PATH"

on_exit() {
  local code=$?
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$FINISHED_AT_PATH"
  echo "$code" > "$EXIT_CODE_PATH"
}
trap on_exit EXIT

if [[ ! -f "$SRC_DIR/$REQUIREMENTS_FILE" ]]; then
  echo "requirements file not found: $SRC_DIR/$REQUIREMENTS_FILE"
  exit 3
fi
if [[ ! -d "$DETECTOR_DIR_REMOTE" ]]; then
  echo "missing detector dir: $DETECTOR_DIR_REMOTE"
  exit 4
fi
if [[ ! -f "$TOKENS_FILE_REMOTE" ]]; then
  echo "missing tokens file: $TOKENS_FILE_REMOTE"
  exit 5
fi
if [[ -z "${!CLICKHOUSE_DSN_ENV:-}" ]]; then
  echo "missing required env var: $CLICKHOUSE_DSN_ENV"
  exit 6
fi

cd "$SRC_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$SRC_DIR/$REQUIREMENTS_FILE"
if [[ -n "$SETUP_COMMAND" ]]; then
  bash -lc "$SETUP_COMMAND"
fi
export PYTHONPATH="$SRC_DIR"
export PYTHONUNBUFFERED=1
echo "$LAUNCH_COMMAND" > "$LAUNCH_COMMAND_PATH"
bash -lc "$LAUNCH_COMMAND"
