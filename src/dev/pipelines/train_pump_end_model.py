import argparse
from datetime import datetime, timedelta

import pandas as pd

from src.shared.pump_end.feature_builder import PumpFeatureBuilder
from src.dev.ml.artifacts import RunArtifacts
from src.dev.ml.dataset import load_labels, build_training_points, deduplicate_points
from src.dev.ml.split import time_split, ratio_split, get_split_info, apply_embargo, clip_points_to_split_bounds
from src.dev.ml.train import train_model, get_feature_columns, get_feature_importance, get_feature_importance_grouped
from src.dev.ml.threshold import threshold_sweep, _prepare_event_data
from src.dev.ml.evaluate import evaluate, evaluate_with_trade_quality
from src.dev.ml.predict import predict_proba, extract_signals
from src.dev.ml.tuning import tune_model, tune_model_both_strategies, train_final_model, get_rule_parameter_grid
from src.shared.clickhouse import DataLoader


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]


def prune_feature_columns(feature_columns: list) -> list:
    prune_prefixes = [
        'liq_sweep_flag_lag_',
        'liq_sweep_overshoot_lag_',
        'liq_reject_strength_lag_',
    ]

    prune_names = {
        'touched_pdh', 'touched_pwh',
        'sweep_pdh', 'sweep_pwh', 'sweep_eqh',
        'overshoot_pdh', 'overshoot_pwh', 'overshoot_eqh',
        'liq_level_type_pwh',
        'vol_spike_cond', 'vol_spike_recent',
        'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'osc_extreme', 'macd_pos_recent',
        'pump_ctx', 'strong_cond', 'pump_score',
        'predump_mask', 'predump_peak',
    }

    pruned = []
    for col in feature_columns:
        if col in prune_names:
            continue
        if any(col.startswith(prefix) for prefix in prune_prefixes):
            continue
        pruned.append(col)

    return pruned


def validate_features_parquet(features_df: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {'event_id', 'offset', 'y'}
    missing_cols = required_cols - set(features_df.columns)
    if missing_cols:
        raise ValueError(f"Features parquet missing required columns: {missing_cols}")

    features_events = set(features_df['event_id'].unique())
    points_events = set(points_df['event_id'].unique())

    common_events = features_events & points_events
    missing_in_features = points_events - features_events
    extra_in_features = features_events - points_events

    if missing_in_features:
        log("WARN", "TRAIN", f"events in points but not in features: {len(missing_in_features)}")
    if extra_in_features:
        log("WARN", "TRAIN", f"events in features but not in points: {len(extra_in_features)}")

    features_df = features_df[features_df['event_id'].isin(common_events)]
    log("INFO", "TRAIN", f"common events after validation: {len(common_events)}")

    return features_df


def check_event_integrity(features_df: pd.DataFrame) -> pd.DataFrame:
    offset_zero = features_df[features_df['offset'] == 0]
    event_counts = offset_zero.groupby('event_id').size()

    valid_events = event_counts[event_counts == 1].index
    invalid_events = event_counts[event_counts != 1].index

    if len(invalid_events) > 0:
        log("WARN", "TRAIN", f"dropping {len(invalid_events)} events with missing/duplicate offset=0")

    features_df = features_df[features_df['event_id'].isin(valid_events)]
    return features_df


def check_nan_features(features_df: pd.DataFrame, feature_columns: list) -> int:
    nan_rows = features_df[feature_columns].isna().any(axis=1).sum()
    total_rows = len(features_df)
    if nan_rows > 0:
        log("WARN", "TRAIN", f"rows with NaN in features: {nan_rows}/{total_rows} ({nan_rows / total_rows * 100:.2f}%)")
    return nan_rows


def filter_features_by_event_time(features_df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    valid_events = event_times[event_times['open_time'] < cutoff]['event_id']
    return features_df[features_df['event_id'].isin(valid_events)].copy()


def calibrate_threshold_on_val(
        model,
        features_df: pd.DataFrame,
        feature_columns: list,
        train_end: datetime,
        val_end: datetime,
        signal_rule: str,
        alpha_hit1: float,
        beta_early: float,
        gamma_miss: float
) -> dict:
    if signal_rule == 'argmax_per_event':
        raise ValueError("argmax_per_event is offline-only and must not be used for threshold calibration.")

    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    val_events = event_times[
        (event_times['open_time'] >= train_end) &
        (event_times['open_time'] < val_end)
        ]['event_id']

    val_df = features_df[
        (features_df['event_id'].isin(val_events)) &
        (features_df['open_time'] >= train_end) &
        (features_df['open_time'] < val_end)
        ].copy()

    if len(val_df) == 0:
        return {'threshold': 0.1, 'min_pending_bars': 1, 'drop_delta': 0.0}

    val_df['split'] = 'val'
    predictions = predict_proba(model, val_df, feature_columns)

    event_data = _prepare_event_data(predictions)
    rule_combinations = get_rule_parameter_grid()

    best_score = -float('inf')
    best_threshold = 0.1
    best_min_pending_bars = 1
    best_drop_delta = 0.0

    for rule_params in rule_combinations:
        min_pending_bars = rule_params['min_pending_bars']
        drop_delta = rule_params['drop_delta']

        threshold, sweep_df = threshold_sweep(
            predictions,
            alpha_hit1=alpha_hit1,
            beta_early=beta_early,
            gamma_miss=gamma_miss,
            signal_rule=signal_rule,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            event_data=event_data
        )

        best_row = sweep_df[sweep_df['threshold'] == threshold].iloc[0]
        score = best_row['score']

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_min_pending_bars = min_pending_bars
            best_drop_delta = drop_delta

    return {
        'threshold': best_threshold,
        'min_pending_bars': best_min_pending_bars,
        'drop_delta': best_drop_delta
    }


def run_build_dataset(args, artifacts: RunArtifacts):
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = parse_date_exclusive(args.end_date) if args.end_date else None

    log("INFO", "BUILD", f"loading labels from {args.labels}")
    labels_df = load_labels(args.labels, start_date, end_date)
    log("INFO", "BUILD",
        f"loaded {len(labels_df)} labels (A={len(labels_df[labels_df['pump_la_type'] == 'A'])}, B={len(labels_df[labels_df['pump_la_type'] == 'B'])})")

    artifacts.save_labels_filtered(labels_df)

    pos_offsets = parse_pos_offsets(args.pos_offsets)
    log("INFO", "BUILD",
        f"building training points neg_before={args.neg_before} neg_after={args.neg_after} pos_offsets={pos_offsets}")
    points_df = build_training_points(
        labels_df,
        neg_before=args.neg_before,
        neg_after=args.neg_after,
        pos_offsets=pos_offsets,
        include_b=args.include_b
    )
    points_df = deduplicate_points(points_df)
    log("INFO", "BUILD",
        f"training points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    artifacts.save_training_points(points_df)

    log("INFO", "BUILD", f"building features from ClickHouse")
    builder = PumpFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=args.window_bars,
        warmup_bars=args.warmup_bars,
        feature_set=args.feature_set
    )

    feature_input = points_df[['symbol', 'open_time']].copy()
    feature_input = feature_input.rename(columns={'open_time': 'event_open_time'})
    feature_input['pump_la_type'] = 'A'
    feature_input['runup_pct'] = 0

    features_df = builder.build(feature_input, max_workers=args.build_workers)

    features_df = features_df.merge(
        points_df[['symbol', 'open_time', 'event_id', 'offset', 'y']],
        on=['symbol', 'open_time'],
        how='inner'
    )

    features_df = check_event_integrity(features_df)
    features_df = features_df.sort_values(['event_id', 'offset']).reset_index(drop=True)

    log("INFO", "BUILD", f"features shape: {features_df.shape}")
    artifacts.save_features(features_df)

    log("INFO", "BUILD", f"dataset saved to {artifacts.get_path()}")


def run_train_only(args, artifacts: RunArtifacts):
    log("INFO", "TRAIN", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TRAIN", f"loaded {len(features_df)} rows with {len(feature_columns)} features")

    if args.prune_features:
        original_count = len(feature_columns)
        feature_columns = prune_feature_columns(feature_columns)
        log("INFO", "TRAIN",
            f"pruned features: {original_count} -> {len(feature_columns)} (removed {original_count - len(feature_columns)})")

    if args.split_strategy == "time":
        train_end = parse_date_exclusive(args.train_end)
        val_end = parse_date_exclusive(args.val_end)
        features_df = time_split(features_df, train_end, val_end)

        if args.embargo_bars > 0:
            features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

        features_df = clip_points_to_split_bounds(features_df, train_end, val_end)
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

    check_nan_features(features_df, feature_columns)

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
    artifacts.save_predictions(val_predictions, 'val')

    log("INFO", "TRAIN", "searching optimal threshold")
    best_threshold, sweep_df = threshold_sweep(
        val_predictions,
        grid_from=args.threshold_grid_from,
        grid_to=args.threshold_grid_to,
        grid_step=args.threshold_grid_step,
        alpha_hit1=args.alpha_hit1,
        beta_early=args.beta_early,
        gamma_miss=args.gamma_miss,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta
    )

    artifacts.save_threshold_sweep(sweep_df)
    artifacts.save_best_threshold(best_threshold, {
        'signal_rule': args.signal_rule,
        'min_pending_bars': args.min_pending_bars,
        'drop_delta': args.drop_delta
    })
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    log("INFO", "TRAIN", "evaluating on val set")
    val_metrics = evaluate(val_predictions, best_threshold, signal_rule=args.signal_rule,
                           min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta)
    artifacts.save_metrics(val_metrics, 'val')
    log("INFO", "TRAIN",
        f"val metrics: hit0={val_metrics['event_level']['hit0_rate']:.3f} early={val_metrics['event_level']['early_rate']:.3f} miss={val_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "predicting on test set")
    test_predictions = predict_proba(
        model,
        features_df[features_df['split'] == 'test'],
        feature_columns
    )
    artifacts.save_predictions(test_predictions, 'test')

    log("INFO", "TRAIN", "evaluating on test set")
    test_metrics = evaluate(test_predictions, best_threshold, signal_rule=args.signal_rule,
                            min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta)
    artifacts.save_metrics(test_metrics, 'test')
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "extracting holdout signals")
    signals_df = extract_signals(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                 min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta)
    artifacts.save_predicted_signals(signals_df)
    log("INFO", "TRAIN", f"saved {len(signals_df)} predicted signals to holdout csv")

    log("INFO", "TRAIN", f"done. artifacts saved to {artifacts.get_path()}")


def run_tune(args, artifacts: RunArtifacts):
    log("INFO", "TUNE", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TUNE", f"loaded {len(features_df)} rows with {len(feature_columns)} features")

    if args.prune_features:
        original_count = len(feature_columns)
        feature_columns = prune_feature_columns(feature_columns)
        log("INFO", "TUNE",
            f"pruned features: {original_count} -> {len(feature_columns)} (removed {original_count - len(feature_columns)})")

    train_end = parse_date_exclusive(args.train_end) if args.train_end else None
    val_end = parse_date_exclusive(args.val_end) if args.val_end else None

    if train_end:
        cv_features_df = filter_features_by_event_time(features_df, train_end)
        log("INFO", "TUNE", f"filtered CV data: {len(cv_features_df)} rows (events before {args.train_end})")
    else:
        cv_features_df = features_df

    log("INFO", "TUNE", f"starting tuning with time_budget={args.time_budget_min}min strategy={args.tune_strategy}")

    if args.tune_strategy == 'both':
        tune_result = tune_model_both_strategies(
            cv_features_df,
            feature_columns,
            time_budget_min=args.time_budget_min,
            fold_months=args.fold_months,
            min_train_months=args.min_train_months,
            signal_rule=args.signal_rule,
            alpha_hit1=args.alpha_hit1,
            beta_early=args.beta_early,
            gamma_miss=args.gamma_miss,
            embargo_bars=args.embargo_bars,
            iterations=args.iterations,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed
        )

        log("INFO", "TUNE", f"winner strategy: {tune_result['winner']}")
        log("INFO", "TUNE", f"threshold score: {tune_result['threshold_result']['best_score']:.4f}")
        log("INFO", "TUNE", f"ranking score: {tune_result['ranking_result']['best_score']:.4f}")

        best_result = tune_result['best_result']
        actual_strategy = tune_result['winner']

        artifacts.save_best_params({
            **best_result['best_params'],
            'tune_strategy': actual_strategy,
            'winner': tune_result['winner'],
            'threshold_score': tune_result['threshold_result']['best_score'],
            'ranking_score': tune_result['ranking_result']['best_score']
        })
        artifacts.save_leaderboard(best_result['leaderboard'])
        artifacts.save_cv_report(best_result['best_cv_result'])
        artifacts.save_folds(best_result['folds'])

    else:
        tune_result = tune_model(
            cv_features_df,
            feature_columns,
            time_budget_min=args.time_budget_min,
            fold_months=args.fold_months,
            min_train_months=args.min_train_months,
            signal_rule=args.signal_rule,
            alpha_hit1=args.alpha_hit1,
            beta_early=args.beta_early,
            gamma_miss=args.gamma_miss,
            embargo_bars=args.embargo_bars,
            iterations=args.iterations,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed,
            tune_strategy=args.tune_strategy
        )

        best_result = tune_result
        actual_strategy = args.tune_strategy

        log("INFO", "TUNE",
            f"tuning completed: {tune_result['trials_completed']} trials in {tune_result['time_elapsed_sec']:.1f}s")
        log("INFO", "TUNE", f"best score: {tune_result['best_score']:.4f}")
        log("INFO", "TUNE", f"best params: {tune_result['best_params']}")

        artifacts.save_best_params({**tune_result['best_params'], 'tune_strategy': actual_strategy})
        artifacts.save_leaderboard(tune_result['leaderboard'])
        artifacts.save_cv_report(tune_result['best_cv_result'])
        artifacts.save_folds(tune_result['folds'])

    actual_signal_rule = args.signal_rule

    if train_end:
        log("INFO", "TUNE", f"training final model on data up to {args.train_end} with strategy={actual_strategy}")

        final_model = train_final_model(
            features_df,
            feature_columns,
            best_result['best_params'],
            train_end,
            iterations=args.iterations,
            seed=args.seed,
            tune_strategy=actual_strategy
        )

        artifacts.save_model(final_model)
        log("INFO", "TUNE", f"final model saved")

        importance_df = get_feature_importance(final_model, feature_columns)
        artifacts.save_feature_importance(importance_df)

        importance_grouped_df = get_feature_importance_grouped(importance_df)
        artifacts.save_feature_importance_grouped(importance_grouped_df)

        if val_end:
            log("INFO", "TUNE", f"calibrating threshold on val window [{args.train_end}, {args.val_end})")

            calibration_result = calibrate_threshold_on_val(
                final_model,
                features_df,
                feature_columns,
                train_end,
                val_end,
                actual_signal_rule,
                args.alpha_hit1,
                args.beta_early,
                args.gamma_miss
            )

            best_threshold = calibration_result['threshold']
            best_min_pending_bars = calibration_result['min_pending_bars']
            best_drop_delta = calibration_result['drop_delta']

            log("INFO", "TUNE",
                f"calibrated: threshold={best_threshold:.3f} min_pending_bars={best_min_pending_bars} drop_delta={best_drop_delta}")

            artifacts.save_best_threshold(best_threshold, {
                'signal_rule': actual_signal_rule,
                'min_pending_bars': best_min_pending_bars,
                'drop_delta': best_drop_delta
            })

            features_df = time_split(features_df, train_end, val_end)

            if args.embargo_bars > 0:
                features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

            features_df = clip_points_to_split_bounds(features_df, train_end, val_end)

            test_predictions = predict_proba(
                final_model,
                features_df[features_df['split'] == 'test'],
                feature_columns
            )
            artifacts.save_predictions(test_predictions, 'test')

            if args.clickhouse_dsn:
                log("INFO", "TUNE", "evaluating with trade quality metrics")
                loader = DataLoader(args.clickhouse_dsn)
                test_metrics = evaluate_with_trade_quality(
                    test_predictions,
                    best_threshold,
                    loader,
                    signal_rule=actual_signal_rule,
                    min_pending_bars=best_min_pending_bars,
                    drop_delta=best_drop_delta,
                    horizons=[16, 32]
                )
                log("INFO", "TUNE",
                    f"trade quality score: {test_metrics['trade_quality_score']:.4f}")
                if 'mfe_short_32' in test_metrics['trade_quality'] and test_metrics['trade_quality']['mfe_short_32']:
                    mfe_stats = test_metrics['trade_quality']['mfe_short_32']
                    log("INFO", "TUNE",
                        f"MFE_32: median={mfe_stats.get('median', 0):.4f} pct_above_2pct={mfe_stats.get('pct_above_2pct', 0):.2f}")
            else:
                test_metrics = evaluate(
                    test_predictions,
                    best_threshold,
                    signal_rule=actual_signal_rule,
                    min_pending_bars=best_min_pending_bars,
                    drop_delta=best_drop_delta
                )

            artifacts.save_metrics(test_metrics, 'test')
            log("INFO", "TUNE",
                f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

            signals_df = extract_signals(
                test_predictions,
                best_threshold,
                signal_rule=actual_signal_rule,
                min_pending_bars=best_min_pending_bars,
                drop_delta=best_drop_delta
            )
            artifacts.save_predicted_signals(signals_df)
            log("INFO", "TUNE", f"saved {len(signals_df)} predicted signals")

    log("INFO", "TUNE", f"done. artifacts saved to {artifacts.get_path()}")


def main():
    parser = argparse.ArgumentParser(description="Train pump end prediction model")

    parser.add_argument("--mode", type=str, choices=["build-dataset", "train", "tune"], required=True)

    parser.add_argument("--labels", type=str)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--dataset-parquet", type=str, default=None)

    parser.add_argument("--neg-before", type=int, default=20)
    parser.add_argument("--neg-after", type=int, default=0)
    parser.add_argument("--pos-offsets", type=str, default="0")
    parser.add_argument("--include-b", action="store_true", default=False)

    parser.add_argument("--window-bars", type=int, default=30)
    parser.add_argument("--warmup-bars", type=int, default=150)
    parser.add_argument("--feature-set", type=str, choices=["base", "extended"], default="base")
    parser.add_argument("--build-workers", type=int, default=4)

    parser.add_argument("--split-strategy", type=str, choices=["time", "ratio"], default="time")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--thread-count", type=int, default=-1)

    parser.add_argument("--threshold-grid-from", type=float, default=0.01)
    parser.add_argument("--threshold-grid-to", type=float, default=0.30)
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)
    parser.add_argument("--alpha-hit1", type=float, default=0.5)
    parser.add_argument("--beta-early", type=float, default=2.0)
    parser.add_argument("--gamma-miss", type=float, default=1.0)

    parser.add_argument("--signal-rule", type=str, choices=["first_cross", "pending_turn_down", "argmax_per_event"],
                        default="pending_turn_down")
    parser.add_argument("--min-pending-bars", type=int, default=1)
    parser.add_argument("--drop-delta", type=float, default=0.0)

    parser.add_argument("--tune-strategy", type=str, choices=["threshold", "ranking", "both"],
                        default="threshold")
    parser.add_argument("--time-budget-min", type=int, default=60)
    parser.add_argument("--fold-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=3)

    parser.add_argument("--prune-features", action="store_true", default=False)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.mode in ("train", "tune") and args.signal_rule == "argmax_per_event":
        parser.error(
            "--signal-rule argmax_per_event is offline-only and non-causal. Use pending_turn_down or first_cross.")

    if args.mode == "build-dataset":
        if not args.labels or not args.clickhouse_dsn:
            parser.error("--labels and --clickhouse-dsn required for build-dataset mode")

    if args.mode == "train":
        if not args.dataset_parquet:
            parser.error("--dataset-parquet required for train mode")
        if args.split_strategy == "time" and (not args.train_end or not args.val_end):
            parser.error("--train-end and --val-end required for time split strategy")

    if args.mode == "tune":
        if not args.dataset_parquet:
            parser.error("--dataset-parquet required for tune mode")

    artifacts = RunArtifacts(args.out_dir, args.run_name)
    log("INFO", "MAIN", f"run_dir={artifacts.get_path()}")

    config = vars(args)
    artifacts.save_config(config)

    if args.mode == "build-dataset":
        run_build_dataset(args, artifacts)
    elif args.mode == "train":
        run_train_only(args, artifacts)
    elif args.mode == "tune":
        run_tune(args, artifacts)


if __name__ == "__main__":
    main()
