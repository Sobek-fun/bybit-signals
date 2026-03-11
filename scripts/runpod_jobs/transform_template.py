from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    df = pd.read_parquet(run_root / "regime_dataset_base.parquet")
    df.to_parquet(run_root / "regime_dataset_train.parquet", index=False)
    target_col = "target_pause_value_next_12h"
    (run_root / "target_col.txt").write_text(target_col, encoding="utf-8")
    print(target_col)


if __name__ == "__main__":
    main()
