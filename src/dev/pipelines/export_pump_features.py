import argparse
from datetime import datetime

import pandas as pd

from src.shared.pump_end.feature_builder import PumpFeatureBuilder


def main():
    parser = argparse.ArgumentParser(description="Export pump feature dataset")
    parser.add_argument(
        "--clickhouse-dsn",
        type=str,
        required=True,
        help="ClickHouse DSN (e.g., http://user:pass@host:port/database)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels CSV file (output of export_pump_labels)"
    )
    parser.add_argument(
        "--window-bars",
        type=int,
        default=30,
        help="Number of bars in feature window (default: 30)"
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=150,
        help="Number of bars for indicator warmup (default: 150)"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["base", "extended"],
        default="base",
        help="Feature set to use: base or extended (default: base)"
    )

    args = parser.parse_args()

    print(f"Loading labels from {args.labels}...")
    labels_df = pd.read_csv(args.labels)

    print(f"Found {len(labels_df)} labels")
    print(
        f"Building features with window_bars={args.window_bars}, warmup_bars={args.warmup_bars}, feature_set={args.feature_set}...")

    builder = PumpFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=args.window_bars,
        warmup_bars=args.warmup_bars,
        feature_set=args.feature_set
    )

    features_df = builder.build(labels_df)

    if features_df.empty:
        print("No features extracted")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"pump_features_tf15m_w{args.window_bars}_warm{args.warmup_bars}_{args.feature_set}_{timestamp}.parquet"

    features_df.to_parquet(output_filename, index=False)

    a_count = len(features_df[features_df['target'] == 1])
    b_count = len(features_df[features_df['target'] == 0])

    print(f"Exported {len(features_df)} rows (A={a_count}, B={b_count}) to {output_filename}")


if __name__ == "__main__":
    main()
