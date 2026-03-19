import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pump_end_threshold.features.feature_builder import PumpFeatureBuilder
from pump_end_threshold.ml.artifacts import RunArtifacts
from pump_end_threshold.ml.dataset import load_labels, build_training_points, deduplicate_points
from pump_end_threshold.ml.split import time_split, ratio_split, get_split_info, apply_embargo, clip_points_to_split_bounds
from pump_end_threshold.ml.train import train_model, get_feature_columns, get_feature_importance, get_feature_importance_grouped
from pump_end_threshold.ml.threshold import threshold_sweep, _prepare_event_data
from pump_end_threshold.ml.evaluate import (
    evaluate_with_trade_quality,
    attach_signal_quality_columns,
    compute_signal_quality_metrics,
)
from pump_end_threshold.ml.predict import predict_proba, extract_signals_verbose
from pump_end_threshold.ml.tuning import (
    tune_model, train_final_model, get_rule_parameter_grid,
    generate_walk_forward_folds, apply_fold_split, apply_fold_embargo,
    clip_fold_points, train_fold, evaluate_fold,
)
from pump_end_threshold.ml.feature_schema import prune_feature_columns
from pump_end_threshold.infra.clickhouse import DataLoader


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]


def _normalize_symbol(token: str) -> str:
    t = token.strip().upper()
    if not t:
        return ""
    if t.endswith("USDT"):
        return t
    return f"{t}USDT"


def _parse_symbols_from_csv(raw: str) -> list[str]:
    out = []
    for part in raw.split(","):
        symbol = _normalize_symbol(part)
        if symbol:
            out.append(symbol)
    return list(dict.fromkeys(out))


def load_symbols_from_file(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"symbols file not found: {path}")
    payload = file_path.read_text(encoding="utf-8")
    normalized = payload.replace("\n", ",")
    return _parse_symbols_from_csv(normalized)


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
        gamma_miss: float,
        threshold_grid_from: float = 0.01,
        threshold_grid_to: float = 0.30,
        threshold_grid_step: float = 0.01,
        delta_fp_b: float = 3.0,
        abstain_margin: float = 0.0,
        clickhouse_dsn: str | None = None,
        quality_density_mode: str = "raw_count",
        quality_target_min_30d: float = 30.0,
        quality_target_max_30d: float = 150.0,
        quality_overflow_penalty: float = 0.03,
        quality_top_k: int = 8,
        quality_entry_shift_bars: int = 0,
) -> dict:
    t0 = time.perf_counter()
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
    val_days = max(1.0, (val_end - train_end).total_seconds() / 86400.0)

    if len(val_df) == 0:
        return {
            'threshold': 0.1,
            'min_pending_bars': 1,
            'drop_delta': 0.0,
            'calibration_sweep': pd.DataFrame(),
            'val_predictions': pd.DataFrame(),
        }

    val_df['split'] = 'val'
    predictions = predict_proba(model, val_df, feature_columns)

    event_data = _prepare_event_data(predictions)
    rule_combinations = get_rule_parameter_grid()
    loader = DataLoader(clickhouse_dsn) if clickhouse_dsn else None

    best_score = -float('inf')
    best_quality_score = -float('inf')
    best_threshold = 0.1
    best_min_pending_bars = 1
    best_drop_delta = 0.0
    rule_payloads = []

    for rule_params in rule_combinations:
        min_pending_bars = rule_params['min_pending_bars']
        drop_delta = rule_params['drop_delta']

        threshold, sweep_df = threshold_sweep(
            predictions,
            grid_from=threshold_grid_from,
            grid_to=threshold_grid_to,
            grid_step=threshold_grid_step,
            alpha_hit1=alpha_hit1,
            beta_early=beta_early,
            gamma_miss=gamma_miss,
            delta_fp_b=delta_fp_b,
            signal_rule=signal_rule,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            event_data=event_data,
            abstain_margin=abstain_margin
        )

        best_row = sweep_df[sweep_df['threshold'] == threshold].iloc[0]
        score = best_row['score']

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_min_pending_bars = min_pending_bars
            best_drop_delta = drop_delta

        sweep_with_rule = sweep_df.copy()
        sweep_with_rule['min_pending_bars'] = min_pending_bars
        sweep_with_rule['drop_delta'] = drop_delta
        sweep_with_rule['quality_score'] = pd.NA
        sweep_with_rule['clean_2_3_share_h32'] = pd.NA
        sweep_with_rule['clean_retrace_precision_2_3_h32'] = pd.NA
        sweep_with_rule['pullback_before_squeeze_share_h32'] = pd.NA
        sweep_with_rule['net_edge_median_h32'] = pd.NA
        sweep_with_rule['pullback_median_h32'] = pd.NA
        sweep_with_rule['squeeze_p75_h32'] = pd.NA
        sweep_with_rule['dirty_no_pullback_2_3_share_h32'] = pd.NA
        sweep_with_rule['signal_count_quality'] = pd.NA
        sweep_with_rule['signals_per_30d_quality'] = pd.NA

        candidate_signals = {}
        if loader is not None:
            candidates_df = sweep_df[sweep_df['signal_count'] > 0].copy()
            if not candidates_df.empty:
                candidates_df['prefilter_score'] = candidates_df['score'] + 0.002 * candidates_df['signal_count']
                candidates_df = candidates_df.sort_values('prefilter_score', ascending=False)
                if quality_top_k > 0:
                    candidates_df = candidates_df.head(quality_top_k)
                for _, candidate in candidates_df.iterrows():
                    candidate_threshold = float(candidate['threshold'])
                    signals_df = extract_signals_verbose(
                        predictions,
                        candidate_threshold,
                        signal_rule=signal_rule,
                        min_pending_bars=min_pending_bars,
                        drop_delta=drop_delta,
                        abstain_margin=abstain_margin
                    )
                    if signals_df.empty:
                        continue
                    candidate_signals[candidate_threshold] = signals_df.copy()

        rule_payloads.append({
            'min_pending_bars': min_pending_bars,
            'drop_delta': drop_delta,
            'sweep_with_rule': sweep_with_rule,
            'candidate_signals': candidate_signals,
        })

    if loader is not None:
        all_signal_frames = []
        for payload in rule_payloads:
            for signals_df in payload['candidate_signals'].values():
                all_signal_frames.append(signals_df)

        if all_signal_frames:
            sample_df = all_signal_frames[0]
            key_columns = [
                c for c in ['event_id', 'symbol', 'open_time', 'event_type', 'signal_offset']
                if c in sample_df.columns
            ]
            if not key_columns:
                key_columns = ['symbol', 'open_time']

            combined_signals = pd.concat(all_signal_frames, ignore_index=True)
            combined_signals = combined_signals.drop_duplicates(subset=key_columns)
            combined_quality = attach_signal_quality_columns(
                combined_signals,
                loader,
                horizons=[32],
                entry_shift_bars=quality_entry_shift_bars,
            )
            combined_quality['__signal_key'] = (
                combined_quality[key_columns].astype(str).agg('|'.join, axis=1)
            )
            combined_quality = combined_quality.drop_duplicates(subset='__signal_key')

            for payload in rule_payloads:
                min_pending_bars = payload['min_pending_bars']
                drop_delta = payload['drop_delta']
                quality_rows = []
                for candidate_threshold, signals_df in payload['candidate_signals'].items():
                    signals_with_keys = signals_df.copy()
                    signals_with_keys['__signal_key'] = (
                        signals_with_keys[key_columns].astype(str).agg('|'.join, axis=1)
                    )
                    candidate_quality_df = combined_quality[
                        combined_quality['__signal_key'].isin(set(signals_with_keys['__signal_key']))
                    ].copy()
                    if candidate_quality_df.empty:
                        continue

                    sq = compute_signal_quality_metrics(
                        signal_path_df=candidate_quality_df,
                        horizon=32,
                        squeeze_threshold=0.02,
                        pullback_threshold=0.03
                    )

                    def _metric(name: str) -> float:
                        value = sq.get(name, 0.0)
                        if pd.isna(value):
                            return 0.0
                        return float(value)

                    clean_2_3_share_h32 = _metric('clean_2_3_share_h32')
                    clean_retrace_precision_2_3_h32 = _metric('clean_retrace_precision_2_3_h32')
                    pullback_before_squeeze_share_2_3_h32 = _metric('pullback_before_squeeze_share_2_3_h32')
                    net_edge_median_h32 = _metric('net_edge_median_h32')
                    pullback_median_h32 = _metric('pullback_median_h32')
                    squeeze_p75_h32 = _metric('squeeze_p75_h32')
                    dirty_no_pullback_2_3_share_h32 = _metric('dirty_no_pullback_2_3_share_h32')
                    signal_count_quality = int(len(signals_df))
                    signals_per_30d_quality = 30.0 * signal_count_quality / val_days

                    if quality_density_mode == "per30d":
                        density_reward = 0.01 * min(signals_per_30d_quality, 120.0)
                        density_under_penalty = 0.18 * max(0.0, quality_target_min_30d - signals_per_30d_quality)
                        density_over_penalty = quality_overflow_penalty * max(
                            0.0, signals_per_30d_quality - quality_target_max_30d
                        )
                    else:
                        density_reward = 0.01 * min(signal_count_quality, 120)
                        density_under_penalty = 0.18 * max(0, 30 - signal_count_quality)
                        density_over_penalty = 0.0

                    quality_score = (
                        5.0 * clean_2_3_share_h32
                        + 3.0 * clean_retrace_precision_2_3_h32
                        + 2.0 * pullback_before_squeeze_share_2_3_h32
                        + 20.0 * net_edge_median_h32
                        + 8.0 * pullback_median_h32
                        - 18.0 * squeeze_p75_h32
                        - 5.0 * dirty_no_pullback_2_3_share_h32
                        + density_reward
                        - density_under_penalty
                        - density_over_penalty
                    )

                    quality_rows.append({
                        'threshold': candidate_threshold,
                        'quality_score': quality_score,
                        'clean_2_3_share_h32': clean_2_3_share_h32,
                        'clean_retrace_precision_2_3_h32': clean_retrace_precision_2_3_h32,
                        'pullback_before_squeeze_share_h32': pullback_before_squeeze_share_2_3_h32,
                        'net_edge_median_h32': net_edge_median_h32,
                        'pullback_median_h32': pullback_median_h32,
                        'squeeze_p75_h32': squeeze_p75_h32,
                        'dirty_no_pullback_2_3_share_h32': dirty_no_pullback_2_3_share_h32,
                        'signal_count_quality': signal_count_quality,
                        'signals_per_30d_quality': signals_per_30d_quality,
                    })

                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_threshold = candidate_threshold
                        best_min_pending_bars = min_pending_bars
                        best_drop_delta = drop_delta

                if quality_rows:
                    quality_df = pd.DataFrame(quality_rows)
                    sweep_with_rule = payload['sweep_with_rule'].merge(
                        quality_df,
                        on='threshold',
                        how='left',
                        suffixes=('', '_quality'),
                    )
                    for col in [
                        'quality_score',
                        'clean_2_3_share_h32',
                        'clean_retrace_precision_2_3_h32',
                        'pullback_before_squeeze_share_h32',
                        'net_edge_median_h32',
                        'pullback_median_h32',
                        'squeeze_p75_h32',
                        'dirty_no_pullback_2_3_share_h32',
                        'signal_count_quality',
                        'signals_per_30d_quality',
                    ]:
                        q_col = f"{col}_quality"
                        if q_col in sweep_with_rule.columns:
                            sweep_with_rule[col] = sweep_with_rule[q_col].where(
                                sweep_with_rule[q_col].notna(),
                                sweep_with_rule[col],
                            )
                            sweep_with_rule = sweep_with_rule.drop(columns=[q_col])
                    payload['sweep_with_rule'] = sweep_with_rule

    all_sweeps = [payload['sweep_with_rule'] for payload in rule_payloads]
    log("INFO", "TUNE", f"calibration took={time.perf_counter() - t0:.2f}s")

    return {
        'threshold': best_threshold,
        'min_pending_bars': best_min_pending_bars,
        'drop_delta': best_drop_delta,
        'calibration_sweep': pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame(),
        'val_predictions': predictions
    }


def run_build_dataset(args, artifacts: RunArtifacts):
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = parse_date_exclusive(args.end_date) if args.end_date else None

    log("INFO", "BUILD", f"loading labels from {args.labels}")
    labels_df = load_labels(args.labels, start_date, end_date)
    if args.symbols_file:
        allowed_symbols = set(load_symbols_from_file(args.symbols_file))
        before = len(labels_df)
        labels_df = labels_df[labels_df['symbol'].astype(str).str.upper().isin(allowed_symbols)].reset_index(drop=True)
        log("INFO", "BUILD", f"filtered labels by symbols-file: {before} -> {len(labels_df)}")
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
        points_df[['symbol', 'open_time', 'event_id', 'offset', 'y', 'pump_la_type', 'runup_pct']],
        on=['symbol', 'open_time'],
        how='inner',
        suffixes=('_feat', '')
    )
    if 'pump_la_type_feat' in features_df.columns:
        features_df = features_df.drop(columns=['pump_la_type_feat'])
    if 'runup_pct_feat' in features_df.columns:
        features_df = features_df.drop(columns=['runup_pct_feat'])

    features_df = check_event_integrity(features_df)
    features_df = features_df.sort_values(['event_id', 'offset']).reset_index(drop=True)

    features_df['sample_weight'] = 1.0

    b_offset0_mask = (features_df['pump_la_type'] == 'B') & (features_df['offset'] == 0)
    features_df.loc[b_offset0_mask, 'sample_weight'] *= 10.0

    if 'pump_ctx' in features_df.columns:
        neg_pump_ctx_mask = (features_df['y'] == 0) & (features_df['pump_ctx'] == 1)
        features_df.loc[neg_pump_ctx_mask, 'sample_weight'] *= 2.0

        neg_no_pump_ctx_mask = (features_df['y'] == 0) & (features_df['pump_ctx'] == 0)
        features_df.loc[neg_no_pump_ctx_mask, 'sample_weight'] *= 0.2

    near_peak_mask = features_df['offset'].abs() <= 3
    features_df.loc[near_peak_mask, 'sample_weight'] *= 1.5

    pos_offset0_mask = (features_df['y'] == 1) & (features_df['offset'] == 0)
    features_df.loc[pos_offset0_mask, 'sample_weight'] *= 2.0

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
        test_end = parse_date_exclusive(args.test_end) if args.test_end else None
        features_df = time_split(features_df, train_end, val_end)

        if args.embargo_bars > 0:
            features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

        features_df = clip_points_to_split_bounds(features_df, train_end, val_end, test_end)
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
        delta_fp_b=args.delta_fp_b,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta,
        abstain_margin=args.abstain_margin
    )

    artifacts.save_threshold_sweep(sweep_df)
    artifacts.save_best_threshold(best_threshold, {
        'signal_rule': args.signal_rule,
        'min_pending_bars': args.min_pending_bars,
        'drop_delta': args.drop_delta,
        'abstain_margin': args.abstain_margin
    })
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    loader = DataLoader(args.clickhouse_dsn) if args.clickhouse_dsn else None

    log("INFO", "TRAIN", "evaluating on val set")
    val_metrics = evaluate_with_trade_quality(
        val_predictions,
        best_threshold,
        loader,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta,
        horizons=[16, 32],
        abstain_margin=args.abstain_margin,
        entry_shift_bars=args.quality_entry_shift_bars,
    )
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
    test_metrics = evaluate_with_trade_quality(
        test_predictions,
        best_threshold,
        loader,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta,
        horizons=[16, 32],
        abstain_margin=args.abstain_margin,
        entry_shift_bars=args.quality_entry_shift_bars,
    )
    artifacts.save_metrics(test_metrics, 'test')
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "extracting holdout signals")
    signals_df = extract_signals_verbose(
        test_predictions,
        best_threshold,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta,
        abstain_margin=args.abstain_margin
    )
    if loader is not None and not signals_df.empty:
        signals_df = attach_signal_quality_columns(
            signals_df,
            loader,
            horizons=[16, 32],
            entry_shift_bars=args.quality_entry_shift_bars,
        )
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
    test_end = parse_date_exclusive(args.test_end) if args.test_end else None

    if train_end:
        cv_features_df = filter_features_by_event_time(features_df, train_end)
        log("INFO", "TUNE", f"filtered CV data: {len(cv_features_df)} rows (events before {args.train_end})")
    else:
        cv_features_df = features_df

    log("INFO", "TUNE", f"starting tuning with time_budget={args.time_budget_min}min strategy={args.tune_strategy}")

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
        threshold_grid_from=args.threshold_grid_from,
        threshold_grid_to=args.threshold_grid_to,
        threshold_grid_step=args.threshold_grid_step,
        delta_fp_b=args.delta_fp_b,
        abstain_margin=args.abstain_margin,
        embargo_bars=args.embargo_bars,
        iterations=args.iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        tune_strategy=args.tune_strategy,
        clickhouse_dsn=args.clickhouse_dsn,
        cv_selection_mode=args.cv_selection_mode,
        quality_top_k=args.quality_top_k,
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
                args.gamma_miss,
                threshold_grid_from=args.threshold_grid_from,
                threshold_grid_to=args.threshold_grid_to,
                threshold_grid_step=args.threshold_grid_step,
                delta_fp_b=args.delta_fp_b,
                abstain_margin=args.abstain_margin,
                clickhouse_dsn=args.clickhouse_dsn,
                quality_density_mode=args.quality_density_mode,
                quality_target_min_30d=args.quality_target_min_30d,
                quality_target_max_30d=args.quality_target_max_30d,
                quality_overflow_penalty=args.quality_overflow_penalty,
                quality_top_k=args.quality_top_k,
                quality_entry_shift_bars=args.quality_entry_shift_bars,
            )

            best_threshold = calibration_result['threshold']
            best_min_pending_bars = calibration_result['min_pending_bars']
            best_drop_delta = calibration_result['drop_delta']

            log("INFO", "TUNE",
                f"calibrated: threshold={best_threshold:.3f} min_pending_bars={best_min_pending_bars} drop_delta={best_drop_delta}")

            artifacts.save_best_threshold(best_threshold, {
                'signal_rule': actual_signal_rule,
                'min_pending_bars': best_min_pending_bars,
                'drop_delta': best_drop_delta,
                'abstain_margin': args.abstain_margin
            })

            features_df = time_split(features_df, train_end, val_end)

            if args.embargo_bars > 0:
                features_df = apply_embargo(features_df, train_end, val_end, args.embargo_bars)

            features_df = clip_points_to_split_bounds(features_df, train_end, val_end, test_end)

            val_predictions = predict_proba(
                final_model,
                features_df[features_df['split'] == 'val'],
                feature_columns
            )
            artifacts.save_predictions(val_predictions, 'val')

            test_predictions = predict_proba(
                final_model,
                features_df[features_df['split'] == 'test'],
                feature_columns
            )
            artifacts.save_predictions(test_predictions, 'test')

            loader = DataLoader(args.clickhouse_dsn) if args.clickhouse_dsn else None
            val_metrics = evaluate_with_trade_quality(
                val_predictions,
                best_threshold,
                loader,
                signal_rule=actual_signal_rule,
                min_pending_bars=best_min_pending_bars,
                drop_delta=best_drop_delta,
                horizons=[16, 32],
                abstain_margin=args.abstain_margin,
                entry_shift_bars=args.quality_entry_shift_bars,
            )
            artifacts.save_metrics(val_metrics, 'val')

            test_metrics = evaluate_with_trade_quality(
                test_predictions,
                best_threshold,
                loader,
                signal_rule=actual_signal_rule,
                min_pending_bars=best_min_pending_bars,
                drop_delta=best_drop_delta,
                horizons=[16, 32],
                abstain_margin=args.abstain_margin,
                entry_shift_bars=args.quality_entry_shift_bars,
            )
            if loader is not None:
                log("INFO", "TUNE", f"trade quality score: {test_metrics['trade_quality_score']:.4f}")
                if 'mfe_short_32' in test_metrics['trade_quality'] and test_metrics['trade_quality']['mfe_short_32']:
                    mfe_stats = test_metrics['trade_quality']['mfe_short_32']
                    log("INFO", "TUNE",
                        f"MFE_32: median={mfe_stats.get('median', 0):.4f} pct_above_2pct={mfe_stats.get('pct_above_2pct', 0):.2f}")

            artifacts.save_metrics(test_metrics, 'test')
            log("INFO", "TUNE",
                f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

            val_signals_df = extract_signals_verbose(
                val_predictions,
                best_threshold,
                signal_rule=actual_signal_rule,
                min_pending_bars=best_min_pending_bars,
                drop_delta=best_drop_delta,
                abstain_margin=args.abstain_margin
            )
            if loader is not None and not val_signals_df.empty:
                val_signals_df = attach_signal_quality_columns(
                    val_signals_df,
                    loader,
                    horizons=[16, 32],
                    entry_shift_bars=args.quality_entry_shift_bars,
                )
            artifacts.save_predicted_signals_val(val_signals_df)
            log("INFO", "TUNE", f"saved {len(val_signals_df)} VAL predicted signals")

            test_signals_df = extract_signals_verbose(
                test_predictions,
                best_threshold,
                signal_rule=actual_signal_rule,
                min_pending_bars=best_min_pending_bars,
                drop_delta=best_drop_delta,
                abstain_margin=args.abstain_margin
            )
            if loader is not None and not test_signals_df.empty:
                test_signals_df = attach_signal_quality_columns(
                    test_signals_df,
                    loader,
                    horizons=[16, 32],
                    entry_shift_bars=args.quality_entry_shift_bars,
                )
            artifacts.save_predicted_signals(test_signals_df)
            log("INFO", "TUNE", f"saved {len(test_signals_df)} predicted signals")

    if args.save_oos_signals:
        log("INFO", "TUNE", "collecting OOS verbose signals from walk-forward folds")
        oos_feature_columns = feature_columns
        oos_folds = tune_result['folds']
        best_params = tune_result['best_params']

        best_threshold_val = None
        best_mpb_val = 1
        best_dd_val = 0.0

        if train_end and val_end:
            best_threshold_val = best_threshold
            best_mpb_val = best_min_pending_bars
            best_dd_val = best_drop_delta
        elif tune_result.get('best_cv_result') and tune_result['best_cv_result'].get('mean_threshold'):
            best_threshold_val = tune_result['best_cv_result']['mean_threshold']

        if best_threshold_val is None:
            best_threshold_val = 0.1

        all_oos_signals = []
        for fold_idx, fold in enumerate(oos_folds):
            fold_df = apply_fold_split(cv_features_df if not train_end else features_df, fold)
            fold_df = apply_fold_embargo(fold_df, fold, args.embargo_bars)
            fold_df = clip_fold_points(fold_df, fold)

            fold_model, _ = train_fold(
                fold_df, oos_feature_columns, best_params,
                iterations=args.iterations,
                early_stopping_rounds=args.early_stopping_rounds,
                seed=args.seed,
            )
            if fold_model is None:
                continue

            val_data = fold_df[fold_df['split'] == 'val']
            if val_data.empty:
                continue

            val_preds = predict_proba(fold_model, val_data, oos_feature_columns)
            verbose_sigs = extract_signals_verbose(
                val_preds,
                best_threshold_val,
                min_pending_bars=best_mpb_val,
                drop_delta=best_dd_val,
                abstain_margin=args.abstain_margin,
            )

            if not verbose_sigs.empty:
                verbose_sigs['fold_idx'] = fold_idx
                verbose_sigs['split'] = 'oos'
                all_oos_signals.append(verbose_sigs)

        if all_oos_signals:
            oos_df = pd.concat(all_oos_signals, ignore_index=True)
            oos_path = artifacts.get_path() / "cv_oos_signals_verbose.parquet"
            oos_df.to_parquet(str(oos_path), index=False)
            log("INFO", "TUNE", f"saved {len(oos_df)} OOS verbose signals to {oos_path}")
        else:
            log("WARN", "TUNE", "no OOS signals collected across folds")

    log("INFO", "TUNE", f"done. artifacts saved to {artifacts.get_path()}")


def main():
    parser = argparse.ArgumentParser(description="Train pump end prediction model")

    parser.add_argument("--mode", type=str, choices=["build-dataset", "train", "tune"], required=True)

    parser.add_argument("--labels", type=str)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--dataset-parquet", type=str, default=None)
    parser.add_argument("--symbols-file", type=str, default=None)

    parser.add_argument("--neg-before", type=int, default=20)
    parser.add_argument("--neg-after", type=int, default=16)
    parser.add_argument("--pos-offsets", type=str, default="0")
    parser.add_argument("--include-b", action="store_true", default=False)

    parser.add_argument("--window-bars", type=int, default=30)
    parser.add_argument("--warmup-bars", type=int, default=150)
    parser.add_argument("--feature-set", type=str, choices=["base", "extended"], default="base")
    parser.add_argument("--build-workers", type=int, default=4)

    parser.add_argument("--split-strategy", type=str, choices=["time", "ratio"], default="time")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
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
    parser.add_argument("--delta-fp-b", type=float, default=3.0)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument("--quality-density-mode", type=str, choices=["raw_count", "per30d"], default="raw_count")
    parser.add_argument("--quality-target-min-30d", type=float, default=30.0)
    parser.add_argument("--quality-target-max-30d", type=float, default=150.0)
    parser.add_argument("--quality-overflow-penalty", type=float, default=0.03)
    parser.add_argument("--quality-top-k", type=int, default=8)
    parser.add_argument("--quality-entry-shift-bars", type=int, default=0)
    parser.add_argument("--cv-selection-mode", type=str, choices=["event_score", "quality_score"], default="event_score")

    parser.add_argument("--signal-rule", type=str, choices=["pending_turn_down"],
                        default="pending_turn_down")
    parser.add_argument("--min-pending-bars", type=int, default=1)
    parser.add_argument("--drop-delta", type=float, default=0.0)

    parser.add_argument("--tune-strategy", type=str, choices=["threshold"],
                        default="threshold")
    parser.add_argument("--time-budget-min", type=int, default=60)
    parser.add_argument("--fold-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=3)

    parser.add_argument("--prune-features", action="store_true", default=False)
    parser.add_argument("--save-oos-signals", action="store_true", default=False,
                        help="Save OOS verbose signals from walk-forward CV folds (for regime guard dataset)")

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

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
