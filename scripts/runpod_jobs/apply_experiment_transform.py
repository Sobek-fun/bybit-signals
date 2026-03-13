from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TP_SL_DERIVED_STRAT_COLUMNS = (
    "strat_resolved_sl_rate_last_24h",
    "strat_resolved_pnl_sum_last_24h",
    "strat_prev_closed_sl_streak",
    "strat_prev_closed_tp_streak",
    "strat_last_closed_is_sl",
    "strat_resolved_sl_rate_last_5",
    "strat_resolved_tp_rate_last_5",
    "strat_resolved_pnl_sum_last_5",
)


def _load_df(run_root: Path) -> pd.DataFrame:
    return pd.read_parquet(run_root / "regime_dataset_base.parquet")


def _save_df(df: pd.DataFrame, run_root: Path) -> None:
    dst = run_root / "regime_dataset_train.parquet"
    df.to_parquet(dst, index=False)


def _drop_tp_sl_derived_strat(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    drop_cols = [c for c in TP_SL_DERIVED_STRAT_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore"), len(drop_cols)


def _drop_market_context(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    drop_cols = [c for c in df.columns if c.startswith(("btc_", "eth_", "breadth_", "btc_eth_"))]
    return df.drop(columns=drop_cols, errors="ignore"), len(drop_cols)


def _apply_target_good_expanded(df: pd.DataFrame) -> pd.DataFrame:
    target_col = "target_pause_value_next_12h_v3_good_expanded"
    df[target_col] = np.where(
        df["future_resolved_count_next_12h"] < 4,
        np.nan,
        np.where(
            (df["future_block_value_next_12h"] <= -10.0) & (df["future_sl_rate_next_12h"] >= 0.55),
            1.0,
            np.where(
                (df["future_block_value_next_12h"] >= 0.0) & (df["future_sl_rate_next_12h"] <= 0.50),
                0.0,
                np.nan,
            ),
        ),
    )
    return df


def _apply_target_local_min3(df: pd.DataFrame) -> pd.DataFrame:
    target_col = "target_pause_value_next_12h_v3_local_min3"
    df[target_col] = np.where(
        df["future_resolved_count_next_12h"] < 3,
        np.nan,
        np.where(
            (df["future_block_value_next_12h"] <= -10.0) & (df["future_sl_rate_next_12h"] >= 0.60),
            1.0,
            np.where(
                (df["future_block_value_next_12h"] >= 0.0) & (df["future_sl_rate_next_12h"] <= 0.50),
                0.0,
                np.nan,
            ),
        ),
    )
    return df


def _exp1(run_root: Path) -> str:
    df = _load_df(run_root)
    df, dropped = _drop_tp_sl_derived_strat(df)
    _save_df(df, run_root)
    print(f"saved={run_root / 'regime_dataset_train.parquet'} rows={len(df)} cols={len(df.columns)} dropped={dropped}")
    return "target_pause_value_next_12h"


def _exp2(run_root: Path) -> str:
    df = _load_df(run_root)
    df, d1 = _drop_market_context(df)
    df, d2 = _drop_tp_sl_derived_strat(df)
    _save_df(df, run_root)
    print(f"saved={run_root / 'regime_dataset_train.parquet'} rows={len(df)} cols={len(df.columns)} dropped={d1 + d2}")
    return "target_pause_value_next_12h"


def _exp3(run_root: Path) -> str:
    df = _load_df(run_root)
    df, _ = _drop_tp_sl_derived_strat(df)
    df = _apply_target_good_expanded(df)
    _save_df(df, run_root)
    target = "target_pause_value_next_12h_v3_good_expanded"
    print(
        f"saved={run_root / 'regime_dataset_train.parquet'} rows={len(df)} cols={len(df.columns)} "
        f"valid={int(df[target].notna().sum())} pos={int((df[target] == 1).sum())} neg={int((df[target] == 0).sum())}"
    )
    return target


def _exp4(run_root: Path) -> str:
    df = _load_df(run_root)
    df, _ = _drop_market_context(df)
    df, _ = _drop_tp_sl_derived_strat(df)
    df = _apply_target_local_min3(df)
    _save_df(df, run_root)
    target = "target_pause_value_next_12h_v3_local_min3"
    print(
        f"saved={run_root / 'regime_dataset_train.parquet'} rows={len(df)} cols={len(df.columns)} "
        f"valid={int(df[target].notna().sum())} pos={int((df[target] == 1).sum())} neg={int((df[target] == 0).sum())}"
    )
    return target


TRANSFORMS = {
    "baseline": _exp1,
    "exp1": _exp1,
    "exp2": _exp2,
    "exp3": _exp3,
    "exp4": _exp4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--params-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    params = json.loads(args.params_json or "{}")
    transform = TRANSFORMS.get(args.exp_id, TRANSFORMS["baseline"])
    target_col = transform(run_root)
    if params.get("target_col"):
        target_col = str(params["target_col"])
    (run_root / "target_col.txt").write_text(target_col, encoding="utf-8")
    print(target_col)


if __name__ == "__main__":
    main()
