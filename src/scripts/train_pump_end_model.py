import argparse
from datetime import datetime

import pandas as pd

from src.datasets.pump_feature_builder import PumpFeatureBuilder
from src.model.artifacts import RunArtifacts
from src.model.dataset import load_labels, build_training_points, deduplicate_points
from src.model.split import time_split, ratio_split, get_split_info
from src.model.train import train_model, get_feature_columns, get_feature_importance, get_feature_importance_grouped
from src.model.threshold import threshold_sweep
from src.model.evaluate import evaluate
from src.model.predict import predict_proba, extract_signals


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def main():
    parser = argparse.ArgumentParser(description="Train pump end prediction model")

    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--features-parquet", type=str, default=None)

    parser.add_argument("--neg-before", type=int, default=20)
    parser.add_argument("--neg-after", type=int, default=0)
    parser.add_argument("--include-b", action="store_true", default=False)

    parser.add_argument("--window-bars", type=int, default=30)
    parser.add_argument("--warmup-bars", type=int, default=150)
    parser.add_argument("--feature-set", type=str, choices=["base", "extended"], default="base")

    parser.add_argument("--split-strategy", type=str, choices=["time", "ratio"], default="time")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--thread-count", type=int, default=-1)

    parser.add_argument("--threshold-grid-from", type=float, default=0.05)
    parser.add_argument("--threshold-grid-to", type=float, default=0.95)
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)
    parser.add_argument("--alpha-hit1", type=float, default=0.5)
    parser.add_argument("--beta-early", type=float, default=2.0)
    parser.add_argument("--gamma-miss", type=float, default=1.0)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.split_strategy == "time" and (not args.train_end or not args.val_end):
        parser.error("--train-end and --val-end required for time split strategy")

    if not args.clickhouse_dsn and not args.features_parquet:
        parser.error("Either --clickhouse-dsn or --features-parquet is required")

    artifacts = RunArtifacts(args.out_dir, args.run_name)
    log("INFO", "TRAIN", f"run_dir={artifacts.get_path()}")

    config = vars(args)
    artifacts.save_config(config)

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None

    log("INFO", "TRAIN", f"loading labels from {args.labels}")
    labels_df = load_labels(args.labels, start_date, end_date)
    log("INFO", "TRAIN",
        f"loaded {len(labels_df)} labels (A={len(labels_df[labels_df['pump_la_type'] == 'A'])}, B={len(labels_df[labels_df['pump_la_type'] == 'B'])})")

    artifacts.save_labels_filtered(labels_df)

    log("INFO", "TRAIN", f"building training points neg_before={args.neg_before} neg_after={args.neg_after}")
    points_df = build_training_points(
        labels_df,
        neg_before=args.neg_before,
        neg_after=args.neg_after,
        include_b=args.include_b
    )
    points_df = deduplicate_points(points_df)
    log("INFO", "TRAIN",
        f"training points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    artifacts.save_training_points(points_df)

    if args.features_parquet:
        log("INFO", "TRAIN", f"loading features from {args.features_parquet}")
        features_df = pd.read_parquet(args.features_parquet)
    else:
        log("INFO", "TRAIN", f"building features from ClickHouse")
        builder = PumpFeatureBuilder(
            ch_dsn=args.clickhouse_dsn,
            window_bars=args.window_bars,
            warmup_bars=args.warmup_bars,
            feature_set=args.feature_set
        )

        feature_input = points_df[['symbol', 'close_time']].copy()
        feature_input = feature_input.rename(columns={'close_time': 'timestamp'})
        feature_input['pump_la_type'] = 'A'
        feature_input['runup_pct'] = 0

        features_df = builder.build(feature_input)

        features_df = features_df.merge(
            points_df[['symbol', 'close_time', 'event_id', 'offset', 'y']],
            on=['symbol', 'close_time'],
            how='inner'
        )

    log("INFO", "TRAIN", f"features shape: {features_df.shape}")
    artifacts.save_features(features_df)

    if args.split_strategy == "time":
        train_end = datetime.strptime(args.train_end, '%Y-%m-%d')
        val_end = datetime.strptime(args.val_end, '%Y-%m-%d')
        features_df = time_split(features_df, train_end, val_end)
    else:
        features_df = ratio_split(
            features_df,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )

    split_info = get_split_info(features_df)
    artifacts.save_splits(split_info)
    log("INFO", "TRAIN",
        f"split info: train={split_info['train']['n_events']} val={split_info['val']['n_events']} test={split_info['test']['n_events']} events")

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TRAIN", f"training with {len(feature_columns)} features")

    model = train_model(
        features_df,
        feature_columns,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        thread_count=args.thread_count,
        seed=args.seed
    )

    artifacts.save_model(model)
    log("INFO", "TRAIN", f"model saved")

    importance_df = get_feature_importance(model, feature_columns)
    artifacts.save_feature_importance(importance_df)

    importance_grouped_df = get_feature_importance_grouped(importance_df)
    artifacts.save_feature_importance_grouped(importance_grouped_df)
    log("INFO", "TRAIN", f"top feature groups: {importance_grouped_df.head(5).to_dict('records')}")

    log("INFO", "TRAIN", "predicting on val set")
    val_predictions = predict_proba(
        model,
        features_df[features_df['split'] == 'val'],
        feature_columns
    )

    log("INFO", "TRAIN", "searching optimal threshold")
    best_threshold, sweep_df = threshold_sweep(
        val_predictions,
        grid_from=args.threshold_grid_from,
        grid_to=args.threshold_grid_to,
        grid_step=args.threshold_grid_step,
        alpha_hit1=args.alpha_hit1,
        beta_early=args.beta_early,
        gamma_miss=args.gamma_miss
    )

    artifacts.save_threshold_sweep(sweep_df)
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    log("INFO", "TRAIN", "evaluating on val set")
    val_metrics = evaluate(val_predictions, best_threshold)
    artifacts.save_metrics(val_metrics, 'val')
    log("INFO", "TRAIN",
        f"val metrics: hit0={val_metrics['event_level']['hit0_rate']:.3f} early={val_metrics['event_level']['early_rate']:.3f} miss={val_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "predicting on test set")
    test_predictions = predict_proba(
        model,
        features_df[features_df['split'] == 'test'],
        feature_columns
    )

    log("INFO", "TRAIN", "evaluating on test set")
    test_metrics = evaluate(test_predictions, best_threshold)
    artifacts.save_metrics(test_metrics, 'test')
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "extracting holdout signals")
    signals_df = extract_signals(test_predictions, best_threshold)
    artifacts.save_predicted_signals(signals_df)
    log("INFO", "TRAIN", f"saved {len(signals_df)} predicted signals to holdout csv")

    log("INFO", "TRAIN", f"done. artifacts saved to {artifacts.get_path()}")


if __name__ == "__main__":
    main()
