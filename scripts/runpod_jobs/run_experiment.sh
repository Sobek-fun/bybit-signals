#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?exp id required}"
RUN_ROOT="${2:?run root required}"
REPO_DIR="${3:?repo dir required}"
CH_DB="${4:?clickhouse dsn required}"
DETECTOR_DIR="${5:?detector dir required}"
TOKENS_FILE="${6:?tokens file required}"
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

if [ "$EXP_ID" = "exp1" ]; then
  uv run --python 3.13 python - <<'PY'
import pandas as pd
from pathlib import Path
import os
run_root = Path(os.environ["RUN_ROOT"])
src = run_root / "regime_dataset_base.parquet"
dst = run_root / "regime_dataset_train.parquet"
df = pd.read_parquet(src)
drop_cols = [c for c in df.columns if c.startswith("strat_resolved_") or c.startswith("strat_prev_closed_") or c in {"strat_last_closed_is_sl","strat_last_closed_is_tp"}]
df = df.drop(columns=drop_cols, errors="ignore")
df.to_parquet(dst, index=False)
print(f"saved={dst} rows={len(df)} cols={len(df.columns)} dropped={len(drop_cols)}")
PY
  TARGET_COL="target_pause_value_next_12h"
fi

if [ "$EXP_ID" = "exp2" ]; then
  uv run --python 3.13 python - <<'PY'
import pandas as pd
from pathlib import Path
import os
run_root = Path(os.environ["RUN_ROOT"])
src = run_root / "regime_dataset_base.parquet"
dst = run_root / "regime_dataset_train.parquet"
df = pd.read_parquet(src)
drop_cols = [c for c in df.columns if c.startswith(("btc_","eth_","breadth_","btc_eth_")) or c.startswith("strat_resolved_") or c.startswith("strat_prev_closed_") or c in {"strat_last_closed_is_sl","strat_last_closed_is_tp"}]
df = df.drop(columns=drop_cols, errors="ignore")
df.to_parquet(dst, index=False)
print(f"saved={dst} rows={len(df)} cols={len(df.columns)} dropped={len(drop_cols)}")
PY
  TARGET_COL="target_pause_value_next_12h"
fi

if [ "$EXP_ID" = "exp3" ]; then
  uv run --python 3.13 python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
import os
run_root = Path(os.environ["RUN_ROOT"])
src = run_root / "regime_dataset_base.parquet"
dst = run_root / "regime_dataset_train.parquet"
df = pd.read_parquet(src)
drop_cols = [c for c in df.columns if c.startswith("strat_resolved_") or c.startswith("strat_prev_closed_") or c in {"strat_last_closed_is_sl","strat_last_closed_is_tp"}]
df = df.drop(columns=drop_cols, errors="ignore")
count_col = "future_resolved_count_next_12h"
sl_col = "future_sl_rate_next_12h"
value_col = "future_block_value_next_12h"
target_col = "target_pause_value_next_12h_v3_good_expanded"
df[target_col] = np.where(df[count_col] < 4, np.nan, np.where((df[value_col] <= -10.0) & (df[sl_col] >= 0.55), 1.0, np.where((df[value_col] >= 0.0) & (df[sl_col] <= 0.50), 0.0, np.nan)))
df.to_parquet(dst, index=False)
valid = int(df[target_col].notna().sum())
pos = int((df[target_col] == 1).sum())
neg = int((df[target_col] == 0).sum())
print(f"saved={dst} rows={len(df)} cols={len(df.columns)} valid={valid} pos={pos} neg={neg}")
PY
  TARGET_COL="target_pause_value_next_12h_v3_good_expanded"
fi

if [ "$EXP_ID" = "exp4" ]; then
  uv run --python 3.13 python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
import os
run_root = Path(os.environ["RUN_ROOT"])
src = run_root / "regime_dataset_base.parquet"
dst = run_root / "regime_dataset_train.parquet"
df = pd.read_parquet(src)
drop_cols = [c for c in df.columns if c.startswith(("btc_","eth_","breadth_","btc_eth_")) or c.startswith("strat_resolved_") or c.startswith("strat_prev_closed_") or c in {"strat_last_closed_is_sl","strat_last_closed_is_tp"}]
df = df.drop(columns=drop_cols, errors="ignore")
count_col = "future_resolved_count_next_12h"
sl_col = "future_sl_rate_next_12h"
value_col = "future_block_value_next_12h"
target_col = "target_pause_value_next_12h_v3_local_min3"
df[target_col] = np.where(df[count_col] < 3, np.nan, np.where((df[value_col] <= -10.0) & (df[sl_col] >= 0.60), 1.0, np.where((df[value_col] >= 0.0) & (df[sl_col] <= 0.50), 0.0, np.nan)))
df.to_parquet(dst, index=False)
valid = int(df[target_col].notna().sum())
pos = int((df[target_col] == 1).sum())
neg = int((df[target_col] == 0).sum())
print(f"saved={dst} rows={len(df)} cols={len(df.columns)} valid={valid} pos={pos} neg={neg}")
PY
  TARGET_COL="target_pause_value_next_12h_v3_local_min3"
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
