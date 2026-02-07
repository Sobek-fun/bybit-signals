import argparse
import subprocess
from datetime import datetime, timedelta

import pandas as pd

from pump_end.features.feature_builder import PumpFeatureBuilder
from pump_end.ml.artifacts import RunArtifacts
from pump_end.ml.dataset import load_labels, build_training_points, deduplicate_points
from pump_end.ml.split import time_split, apply_embargo, clip_points_to_split_bounds
from pump_end.ml.train import get_feature_columns, get_feature_importance, get_feature_importance_grouped
from pump_end.ml.threshold import threshold_sweep, _prepare_event_data
from pump_end.ml.evaluate import evaluate
from pump_end.ml.predict import predict_proba, extract_signals
from pump_end.ml.tuning import tune_model, train_final_model, get_rule_parameter_grid
from pump_end.ml.clustering import (
    fit_event_clusterer, assign_event_clusters, get_available_cluster_features,
    compute_cluster_feature_summary, compute_cluster_drift_by_month
)
from pump_end.infra.clickhouse import DataLoader

FEATURE_SET = 'extended'
WINDOW_BARS = 30
WARMUP_BARS = 150
NEG_BEFORE = 60
NEG_AFTER = 16
POS_OFFSETS = [0]
INCLUDE_B = False
FOLD_MONTHS = 1
MIN_TRAIN_MONTHS = 3
TIME_BUDGET_MIN = 120
ITERATIONS = 1000
SEED = 42
EARLY_STOPPING_ROUNDS = 50
EMBARGO_BARS = 0
THRESHOLD_GRID_FROM = 0.01
THRESHOLD_GRID_TO = 0.95
THRESHOLD_GRID_STEP = 0.01
ALPHA_HIT1 = 0.5
ALPHA_HIT_PRE = 0.25
BETA_EARLY = 2.0
GAMMA_MISS = 1.0
KAPPA_EARLY_MAGNITUDE = 0.22
MIN_TRIGGER_RATE = 0.05
MAX_TRIGGER_RATE = 0.12
CLUSTER_K = 5
CLUSTER_N_COMPONENTS = 5
SEL_MIN_EVENTS = 50
SEL_MIN_TRIGGERED_ABS = 10
SEL_MIN_TRIGGERED_FRAC = 0.005
SEL_MIN_PRECISION_TRIGGERED = 0.20
SEL_MAX_EARLY_FAR_SHARE = 0.65
SEL_MIN_INRANGE_RATE = 0.55
SEL_MAX_TAIL_LE_MINUS10_RATE = 0.03
SEL_MIN_MFE32_PCT_ABOVE_2PCT = 0.75
SEL_MAX_MAE16_P75 = 0.40
SEL_FALLBACK_TOPK_IF_EMPTY = 1


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]


def extract_dataset_params(features_df: pd.DataFrame) -> dict:
    params = {}

    params['neg_before'] = abs(int(features_df['offset'].min()))

    max_offset = features_df['offset'].max()
    params['neg_after'] = int(max_offset) if max_offset > 0 else 0

    if 'window_bars' in features_df.columns:
        params['window_bars'] = int(features_df['window_bars'].iloc[0])

    if 'warmup_bars' in features_df.columns:
        params['warmup_bars'] = int(features_df['warmup_bars'].iloc[0])

    extended_cols = {'atr_norm', 'bb_z', 'vwap_dev', 'obv'}
    has_extended = any(col in features_df.columns for col in extended_cols)
    params['feature_set'] = 'extended' if has_extended else 'base'

    if 'pump_la_type' in features_df.columns:
        params['include_b'] = 'B' in features_df['pump_la_type'].values
    else:
        params['include_b'] = False

    return params


def check_event_integrity(features_df: pd.DataFrame) -> pd.DataFrame:
    offset_zero = features_df[features_df['offset'] == 0]
    event_counts = offset_zero.groupby('event_id').size()

    valid_events = event_counts[event_counts == 1].index
    invalid_events = event_counts[event_counts != 1].index

    if len(invalid_events) > 0:
        log("WARN", "TRAIN", f"dropping {len(invalid_events)} events with missing/duplicate offset=0")

    features_df = features_df[features_df['event_id'].isin(valid_events)]
    return features_df


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
        alpha_hit1: float,
        alpha_hit_pre: float,
        beta_early: float,
        gamma_miss: float,
        kappa_early_magnitude: float,
        min_trigger_rate: float,
        max_trigger_rate: float,
        grid_from: float = 0.01,
        grid_to: float = 0.95,
        grid_step: float = 0.01
) -> dict:
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
        return {'threshold': 0.1, 'min_pending_bars': 1, 'drop_delta': 0.0, 'min_pending_peak': 0.0, 'min_turn_down_bars': 1}

    val_df['split'] = 'val'
    predictions = predict_proba(model, val_df, feature_columns)

    event_data = _prepare_event_data(predictions)
    rule_combinations = get_rule_parameter_grid()

    best_score = -float('inf')
    best_threshold = 0.1
    best_min_pending_bars = 1
    best_drop_delta = 0.0
    best_min_pending_peak = 0.0
    best_min_turn_down_bars = 1

    for rule_params in rule_combinations:
        min_turn_down_bars = rule_params['min_turn_down_bars']
        min_pending_bars = rule_params['min_pending_bars']
        drop_delta = rule_params['drop_delta']
        min_pending_peak = rule_params['min_pending_peak']

        threshold, sweep_df = threshold_sweep(
            predictions,
            grid_from=grid_from,
            grid_to=grid_to,
            grid_step=grid_step,
            alpha_hit1=alpha_hit1,
            alpha_hit_pre=alpha_hit_pre,
            beta_early=beta_early,
            gamma_miss=gamma_miss,
            kappa_early_magnitude=kappa_early_magnitude,
            min_pending_bars=min_pending_bars,
            drop_delta=drop_delta,
            min_pending_peak=min_pending_peak,
            min_turn_down_bars=min_turn_down_bars,
            event_data=event_data,
            min_trigger_rate=min_trigger_rate,
            max_trigger_rate=max_trigger_rate
        )

        best_row = sweep_df[sweep_df['threshold'] == threshold].iloc[0]
        score = best_row['score']

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_min_pending_bars = min_pending_bars
            best_drop_delta = drop_delta
            best_min_pending_peak = min_pending_peak
            best_min_turn_down_bars = min_turn_down_bars

    return {
        'threshold': best_threshold,
        'min_pending_bars': best_min_pending_bars,
        'drop_delta': best_drop_delta,
        'min_pending_peak': best_min_pending_peak,
        'min_turn_down_bars': best_min_turn_down_bars
    }


def calibrate_threshold_on_val_by_cluster(
        model,
        features_df: pd.DataFrame,
        feature_columns: list,
        train_end: datetime,
        val_end: datetime,
        alpha_hit1: float,
        alpha_hit_pre: float,
        beta_early: float,
        gamma_miss: float,
        kappa_early_magnitude: float,
        min_trigger_rate: float,
        max_trigger_rate: float,
        grid_from: float = 0.01,
        grid_to: float = 0.95,
        grid_step: float = 0.01
) -> dict:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time', 'cluster_id']].drop_duplicates(
        'event_id')
    val_events = event_times[
        (event_times['open_time'] >= train_end) &
        (event_times['open_time'] < val_end)
        ]

    val_df = features_df[
        (features_df['event_id'].isin(val_events['event_id'])) &
        (features_df['open_time'] >= train_end) &
        (features_df['open_time'] < val_end)
        ].copy()

    if len(val_df) == 0:
        return {}

    val_df['split'] = 'val'
    predictions = predict_proba(model, val_df, feature_columns)

    cluster_ids = sorted(val_events['cluster_id'].dropna().unique())
    thresholds_by_cluster = {}

    for cluster_id in cluster_ids:
        cluster_events = val_events[val_events['cluster_id'] == cluster_id]['event_id']
        cluster_predictions = predictions[predictions['event_id'].isin(cluster_events)]

        if len(cluster_predictions) == 0:
            continue

        n_events = cluster_predictions['event_id'].nunique()
        if n_events < 5:
            continue

        event_data = _prepare_event_data(cluster_predictions)
        rule_combinations = get_rule_parameter_grid()

        best_score = -float('inf')
        best_threshold = 0.1
        best_min_pending_bars = 1
        best_drop_delta = 0.0
        best_min_pending_peak = 0.0
        best_min_turn_down_bars = 1

        for rule_params in rule_combinations:
            min_turn_down_bars = rule_params['min_turn_down_bars']
            min_pending_bars = rule_params['min_pending_bars']
            drop_delta = rule_params['drop_delta']
            min_pending_peak = rule_params['min_pending_peak']

            threshold, sweep_df = threshold_sweep(
                cluster_predictions,
                grid_from=grid_from,
                grid_to=grid_to,
                grid_step=grid_step,
                alpha_hit1=alpha_hit1,
                alpha_hit_pre=alpha_hit_pre,
                beta_early=beta_early,
                gamma_miss=gamma_miss,
                kappa_early_magnitude=kappa_early_magnitude,
                min_pending_bars=min_pending_bars,
                drop_delta=drop_delta,
                min_pending_peak=min_pending_peak,
                min_turn_down_bars=min_turn_down_bars,
                event_data=event_data,
                min_trigger_rate=min_trigger_rate,
                max_trigger_rate=max_trigger_rate
            )

            best_row = sweep_df[sweep_df['threshold'] == threshold].iloc[0]
            score = best_row['score']

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_min_pending_bars = min_pending_bars
                best_drop_delta = drop_delta
                best_min_pending_peak = min_pending_peak
                best_min_turn_down_bars = min_turn_down_bars

        thresholds_by_cluster[int(cluster_id)] = {
            'threshold': best_threshold,
            'min_pending_bars': best_min_pending_bars,
            'drop_delta': best_drop_delta,
            'min_pending_peak': best_min_pending_peak,
            'min_turn_down_bars': best_min_turn_down_bars,
            'n_events': n_events,
            'best_score': best_score
        }

    return thresholds_by_cluster


def evaluate_clusters_selectivity(
        thresholds_by_cluster: dict,
        val_metrics_by_cluster: dict,
        trade_quality_by_cluster: dict = None,
        min_events: int = 50,
        min_triggered_abs: int = 10,
        min_triggered_frac: float = 0.005,
        min_precision_triggered: float = 0.20,
        max_early_far_share: float = 0.65,
        min_inrange_rate: float = 0.55,
        max_tail_le_minus10_rate: float = 0.03,
        min_mfe32_pct_above_2pct: float = 0.75,
        max_mae16_p75: float = 0.40,
        fallback_topk_if_empty: int = 1
) -> tuple:
    from math import ceil
    from pump_end.ml.evaluate import compute_trade_quality_score

    enabled_clusters = []
    selectivity_report = {}

    if trade_quality_by_cluster is None:
        trade_quality_by_cluster = {}

    for cluster_id, params in thresholds_by_cluster.items():
        if cluster_id == 'enabled_clusters':
            continue

        cluster_report = {
            'n_events': params['n_events'],
            'best_score': params['best_score'],
            'enabled': False,
            'reason': None
        }

        if params['n_events'] < min_events:
            cluster_report['reason'] = f'n_events {params["n_events"]} < {min_events}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if params['best_score'] <= -float('inf'):
            cluster_report['reason'] = 'best_score is -inf'
            selectivity_report[cluster_id] = cluster_report
            continue

        if cluster_id not in val_metrics_by_cluster:
            cluster_report['reason'] = 'no val metrics'
            selectivity_report[cluster_id] = cluster_report
            continue

        metrics = val_metrics_by_cluster[cluster_id]
        n_events = metrics['event_level']['n_events']
        miss = metrics['event_level']['miss']
        hit_window = metrics['event_level'].get('hit_window', 0)
        early_far = metrics['event_level'].get('early_far', 0)
        offset_minus2_to_plus2_rate = metrics['event_level'].get('offset_minus2_to_plus2_rate', 0)
        tail_le_minus10_rate = metrics['event_level'].get('tail_le_minus10_rate', 0)

        triggered = n_events - miss
        min_triggered = max(min_triggered_abs, ceil(min_triggered_frac * n_events))
        precision_triggered = hit_window / triggered if triggered > 0 else 0.0
        early_far_share = early_far / triggered if triggered > 0 else 0.0

        cluster_report['triggered'] = triggered
        cluster_report['min_triggered_required'] = min_triggered
        cluster_report['precision_triggered'] = precision_triggered
        cluster_report['hit_window'] = hit_window
        cluster_report['early_far'] = early_far
        cluster_report['early_far_share'] = early_far_share
        cluster_report['offset_minus2_to_plus2_rate'] = offset_minus2_to_plus2_rate
        cluster_report['tail_le_minus10_rate'] = tail_le_minus10_rate
        cluster_report['early_rate'] = metrics['event_level'].get('early_rate', 0)
        cluster_report['miss_rate'] = metrics['event_level'].get('miss_rate', 0)
        cluster_report['median_pred_offset'] = metrics['event_level'].get('median_pred_offset')

        if triggered < min_triggered:
            cluster_report['reason'] = f'triggered {triggered} < {min_triggered}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if precision_triggered < min_precision_triggered:
            cluster_report['reason'] = f'precision_triggered {precision_triggered:.3f} < {min_precision_triggered}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if early_far_share > max_early_far_share:
            cluster_report['reason'] = f'early_far_share {early_far_share:.3f} > {max_early_far_share}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if offset_minus2_to_plus2_rate < min_inrange_rate:
            cluster_report['reason'] = f'offset_minus2_to_plus2_rate {offset_minus2_to_plus2_rate:.3f} < {min_inrange_rate}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if tail_le_minus10_rate > max_tail_le_minus10_rate:
            cluster_report['reason'] = f'tail_le_minus10_rate {tail_le_minus10_rate:.3f} > {max_tail_le_minus10_rate}'
            selectivity_report[cluster_id] = cluster_report
            continue

        if cluster_id in trade_quality_by_cluster:
            tq = trade_quality_by_cluster[cluster_id]
            mfe_32 = tq.get('mfe_short_32', {})
            mae_16 = tq.get('mae_short_16', {})

            if mfe_32:
                cluster_report['mfe_short_32_median'] = mfe_32.get('median')
                cluster_report['mfe_short_32_pct_above_2pct'] = mfe_32.get('pct_above_2pct')
                pct_above = mfe_32.get('pct_above_2pct', 0)
                if pct_above < min_mfe32_pct_above_2pct:
                    cluster_report['reason'] = f'mfe32_pct_above_2pct {pct_above:.3f} < {min_mfe32_pct_above_2pct}'
                    selectivity_report[cluster_id] = cluster_report
                    continue

            if mae_16:
                cluster_report['mae_short_16_p75'] = mae_16.get('p75')
                mae_p75 = mae_16.get('p75', 0)
                if mae_p75 > max_mae16_p75:
                    cluster_report['reason'] = f'mae16_p75 {mae_p75:.4f} > {max_mae16_p75}'
                    selectivity_report[cluster_id] = cluster_report
                    continue

        cluster_report['enabled'] = True
        cluster_report['reason'] = 'passed all checks'
        selectivity_report[cluster_id] = cluster_report
        enabled_clusters.append(cluster_id)

    if not enabled_clusters and fallback_topk_if_empty > 0:
        candidates = []
        for cluster_id, params in thresholds_by_cluster.items():
            if cluster_id == 'enabled_clusters':
                continue
            if params['n_events'] < min_events:
                continue
            if cluster_id not in val_metrics_by_cluster:
                continue
            metrics = val_metrics_by_cluster[cluster_id]
            n_events = metrics['event_level']['n_events']
            triggered = n_events - metrics['event_level']['miss']
            hit_window = metrics['event_level'].get('hit_window', 0)

            min_triggered_required = max(min_triggered_abs, ceil(min_triggered_frac * n_events))
            if triggered < min_triggered_required:
                continue

            if cluster_id in trade_quality_by_cluster:
                tq = trade_quality_by_cluster[cluster_id]
                utility = compute_trade_quality_score(tq, horizon=32)
            else:
                utility = params['best_score']

            candidates.append((cluster_id, hit_window, triggered, utility))

        if candidates:
            candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            for i in range(min(fallback_topk_if_empty, len(candidates))):
                best_cluster_id, best_hit_window, best_triggered, best_utility = candidates[i]
                enabled_clusters.append(best_cluster_id)
                selectivity_report[best_cluster_id]['enabled'] = True
                selectivity_report[best_cluster_id]['reason'] = f'fallback_enable_best_cluster (hit_window={best_hit_window}, triggered={best_triggered}, utility={best_utility:.4f})'

    if enabled_clusters:
        total_val_events = sum(
            val_metrics_by_cluster[cid]['event_level']['n_events']
            for cid in val_metrics_by_cluster
        )
        enabled_events = sum(
            val_metrics_by_cluster[cid]['event_level']['n_events']
            for cid in enabled_clusters
            if cid in val_metrics_by_cluster
        )
        enabled_coverage = enabled_events / total_val_events if total_val_events > 0 else 0.0

        if enabled_coverage < 0.20:
            for cid in enabled_clusters:
                selectivity_report[cid]['enabled'] = False
                selectivity_report[cid]['reason'] = f'enabled_coverage_too_low__fallback_to_global (coverage={enabled_coverage:.3f})'
            enabled_clusters = []

    return enabled_clusters, selectivity_report


def extract_signals_by_cluster(
        predictions_df: pd.DataFrame,
        thresholds_by_cluster: dict,
        enabled_clusters: list,
        features_df: pd.DataFrame
) -> pd.DataFrame:
    event_cluster_map = features_df[features_df['offset'] == 0][['event_id', 'cluster_id']].drop_duplicates('event_id')
    event_cluster_map = dict(zip(event_cluster_map['event_id'], event_cluster_map['cluster_id']))

    predictions_df = predictions_df.copy()
    predictions_df['cluster_id'] = predictions_df['event_id'].map(event_cluster_map)

    all_signals = []

    for cluster_id in enabled_clusters:
        if cluster_id not in thresholds_by_cluster:
            continue

        params = thresholds_by_cluster[cluster_id]
        cluster_preds = predictions_df[predictions_df['cluster_id'] == cluster_id]

        if len(cluster_preds) == 0:
            continue

        signals = extract_signals(
            cluster_preds,
            threshold=params['threshold'],
            min_pending_bars=params['min_pending_bars'],
            drop_delta=params['drop_delta'],
            min_pending_peak=params['min_pending_peak'],
            min_turn_down_bars=params.get('min_turn_down_bars', 1)
        )

        if not signals.empty:
            signals['cluster_id'] = cluster_id
            all_signals.append(signals)

    if not all_signals:
        return pd.DataFrame(columns=['symbol', 'open_time', 'cluster_id'])

    return pd.concat(all_signals, ignore_index=True)


def evaluate_by_cluster(
        predictions_df: pd.DataFrame,
        thresholds_by_cluster: dict,
        features_df: pd.DataFrame
) -> dict:
    event_cluster_map = features_df[features_df['offset'] == 0][['event_id', 'cluster_id']].drop_duplicates('event_id')
    event_cluster_map = dict(zip(event_cluster_map['event_id'], event_cluster_map['cluster_id']))

    predictions_df = predictions_df.copy()
    predictions_df['cluster_id'] = predictions_df['event_id'].map(event_cluster_map)

    metrics_by_cluster = {}

    for cluster_id, params in thresholds_by_cluster.items():
        if cluster_id == 'enabled_clusters':
            continue

        cluster_preds = predictions_df[predictions_df['cluster_id'] == cluster_id]

        if len(cluster_preds) == 0:
            continue

        metrics = evaluate(
            cluster_preds,
            threshold=params['threshold'],
            min_pending_bars=params['min_pending_bars'],
            drop_delta=params['drop_delta'],
            min_pending_peak=params['min_pending_peak'],
            min_turn_down_bars=params.get('min_turn_down_bars', 1)
        )

        metrics_by_cluster[int(cluster_id)] = metrics

    return metrics_by_cluster


def run_build_dataset(args, artifacts: RunArtifacts):
    config = vars(args).copy()
    artifacts.save_config(config)

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = parse_date_exclusive(args.end_date) if args.end_date else None

    log("INFO", "BUILD", f"loading labels from {args.labels}")
    labels_df = load_labels(args.labels, start_date, end_date)
    log("INFO", "BUILD",
        f"loaded {len(labels_df)} labels (A={len(labels_df[labels_df['pump_la_type'] == 'A'])}, B={len(labels_df[labels_df['pump_la_type'] == 'B'])})")

    artifacts.save_labels_filtered(labels_df)

    log("INFO", "BUILD",
        f"building training points neg_before={NEG_BEFORE} neg_after={NEG_AFTER} pos_offsets={POS_OFFSETS}")
    points_df = build_training_points(
        labels_df,
        neg_before=NEG_BEFORE,
        neg_after=NEG_AFTER,
        pos_offsets=POS_OFFSETS,
        include_b=INCLUDE_B
    )
    points_df = deduplicate_points(points_df)
    log("INFO", "BUILD",
        f"training points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    artifacts.save_training_points(points_df)

    log("INFO", "BUILD", f"building features from ClickHouse")
    builder = PumpFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=WINDOW_BARS,
        warmup_bars=WARMUP_BARS,
        feature_set=FEATURE_SET
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


def run_tune(args, artifacts: RunArtifacts):
    log("INFO", "TUNE", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    config = vars(args).copy()
    dataset_params = extract_dataset_params(features_df)
    config.update(dataset_params)
    artifacts.save_config(config)

    feature_columns = get_feature_columns(features_df)
    log("INFO", "TUNE", f"loaded {len(features_df)} rows with {len(feature_columns)} features")

    train_end = parse_date_exclusive(args.train_end) if args.train_end else None
    val_end = parse_date_exclusive(args.val_end) if args.val_end else None
    test_end = parse_date_exclusive(args.test_end) if args.test_end else None

    if train_end:
        cv_features_df = filter_features_by_event_time(features_df, train_end)
        log("INFO", "TUNE", f"filtered CV data: {len(cv_features_df)} rows (events before {args.train_end})")
    else:
        cv_features_df = features_df

    log("INFO", "TUNE", f"starting tuning with time_budget={TIME_BUDGET_MIN}min")

    tune_result = tune_model(
        cv_features_df,
        feature_columns,
        time_budget_min=TIME_BUDGET_MIN,
        fold_months=FOLD_MONTHS,
        min_train_months=MIN_TRAIN_MONTHS,
        alpha_hit1=ALPHA_HIT1,
        alpha_hit_pre=ALPHA_HIT_PRE,
        beta_early=BETA_EARLY,
        gamma_miss=GAMMA_MISS,
        kappa_early_magnitude=KAPPA_EARLY_MAGNITUDE,
        min_trigger_rate=MIN_TRIGGER_RATE,
        max_trigger_rate=MAX_TRIGGER_RATE,
        embargo_bars=EMBARGO_BARS,
        iterations=ITERATIONS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=SEED
    )

    log("INFO", "TUNE",
        f"tuning completed: {tune_result['trials_completed']} trials in {tune_result['time_elapsed_sec']:.1f}s")
    log("INFO", "TUNE", f"best score: {tune_result['best_score']:.4f}")
    log("INFO", "TUNE", f"best params: {tune_result['best_params']}")

    artifacts.save_best_params(tune_result['best_params'])
    artifacts.save_leaderboard(tune_result['leaderboard'])
    artifacts.save_cv_report(tune_result['best_cv_result'])
    artifacts.save_folds(tune_result['folds'])

    if train_end:
        log("INFO", "TUNE", f"training final model on data up to {args.train_end}")

        final_model = train_final_model(
            features_df,
            feature_columns,
            tune_result['best_params'],
            train_end,
            iterations=ITERATIONS,
            seed=SEED
        )

        artifacts.save_model(final_model)
        log("INFO", "TUNE", f"final model saved")

        importance_df = get_feature_importance(final_model, feature_columns)
        artifacts.save_feature_importance(importance_df)

        importance_grouped_df = get_feature_importance_grouped(importance_df)
        artifacts.save_feature_importance_grouped(importance_grouped_df)

        if val_end:
            log("INFO", "CLUSTER", f"fitting event clusterer with k={CLUSTER_K}")

            cluster_features = get_available_cluster_features(features_df)
            log("INFO", "CLUSTER", f"available cluster features: {len(cluster_features)}")

            clusterer, train_event_clusters, cluster_reports = fit_event_clusterer(
                features_df,
                train_end,
                cluster_features=cluster_features,
                k=CLUSTER_K,
                n_components=CLUSTER_N_COMPONENTS,
                random_state=SEED
            )

            artifacts.save_cluster_model(clusterer)
            artifacts.save_cluster_quality(cluster_reports)

            if cluster_reports.get('cluster_features_dropped'):
                artifacts.save_cluster_features_dropped(cluster_reports['cluster_features_dropped'])

            cluster_features_used = cluster_reports.get('cluster_features_used', cluster_features)

            cluster_config = {
                'cluster_features': cluster_features_used,
                'k': CLUSTER_K,
                'k_used': cluster_reports.get('k_used', CLUSTER_K),
                'n_components': CLUSTER_N_COMPONENTS,
                'train_end': str(train_end),
                'n_train_events': cluster_reports['n_train_events']
            }
            artifacts.save_cluster_config(cluster_config)

            log("INFO", "CLUSTER",
                f"cluster quality: silhouette={cluster_reports['silhouette']:.3f} sizes={cluster_reports['cluster_sizes']}")

            features_df = assign_event_clusters(clusterer, features_df, cluster_features_used)

            cluster_feature_summary = compute_cluster_feature_summary(features_df, cluster_features_used)
            artifacts.save_cluster_feature_summary(cluster_feature_summary)

            cluster_drift = compute_cluster_drift_by_month(features_df)
            artifacts.save_cluster_drift_by_month(cluster_drift)

            log("INFO", "CLUSTER",
                f"calibrating thresholds by cluster on val window [{args.train_end}, {args.val_end})")

            thresholds_by_cluster = calibrate_threshold_on_val_by_cluster(
                final_model,
                features_df,
                feature_columns,
                train_end,
                val_end,
                ALPHA_HIT1,
                ALPHA_HIT_PRE,
                BETA_EARLY,
                GAMMA_MISS,
                KAPPA_EARLY_MAGNITUDE,
                MIN_TRIGGER_RATE,
                MAX_TRIGGER_RATE,
                grid_from=THRESHOLD_GRID_FROM,
                grid_to=THRESHOLD_GRID_TO,
                grid_step=THRESHOLD_GRID_STEP
            )

            global_calibration = calibrate_threshold_on_val(
                final_model,
                features_df,
                feature_columns,
                train_end,
                val_end,
                ALPHA_HIT1,
                ALPHA_HIT_PRE,
                BETA_EARLY,
                GAMMA_MISS,
                KAPPA_EARLY_MAGNITUDE,
                MIN_TRIGGER_RATE,
                MAX_TRIGGER_RATE,
                grid_from=THRESHOLD_GRID_FROM,
                grid_to=THRESHOLD_GRID_TO,
                grid_step=THRESHOLD_GRID_STEP
            )
            log("INFO", "CLUSTER", f"global fallback calibration: threshold={global_calibration['threshold']:.3f}")

            artifacts.save_best_threshold(global_calibration['threshold'], {
                'min_pending_bars': global_calibration['min_pending_bars'],
                'drop_delta': global_calibration['drop_delta'],
                'min_pending_peak': global_calibration['min_pending_peak'],
                'min_turn_down_bars': global_calibration['min_turn_down_bars'],
                'note': 'global_fallback_for_prod_and_empty_clusters'
            })

            features_df = time_split(features_df, train_end, val_end)

            if EMBARGO_BARS > 0:
                features_df = apply_embargo(features_df, train_end, val_end, EMBARGO_BARS)

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

            all_cluster_ids = sorted([k for k in thresholds_by_cluster.keys() if k != 'enabled_clusters'])
            val_signals_raw = extract_signals_by_cluster(val_predictions, thresholds_by_cluster, all_cluster_ids, features_df)
            test_signals_raw = extract_signals_by_cluster(test_predictions, thresholds_by_cluster, all_cluster_ids, features_df)
            artifacts.save_predicted_signals_raw(val_signals_raw, 'val')
            artifacts.save_predicted_signals_raw(test_signals_raw, 'test')
            log("INFO", "CLUSTER", f"saved raw signals: val={len(val_signals_raw)} test={len(test_signals_raw)}")

            val_metrics_by_cluster = evaluate_by_cluster(val_predictions, thresholds_by_cluster, features_df)
            artifacts.save_metrics_by_cluster(val_metrics_by_cluster, 'val')

            trade_quality_by_cluster = {}
            if args.clickhouse_dsn:
                from pump_end.ml.evaluate import compute_trade_quality_metrics
                loader = DataLoader(args.clickhouse_dsn)
                for cluster_id in thresholds_by_cluster.keys():
                    if cluster_id == 'enabled_clusters':
                        continue
                    params = thresholds_by_cluster[cluster_id]
                    cluster_preds = val_predictions[val_predictions['event_id'].isin(
                        features_df[(features_df['offset'] == 0) & (features_df['cluster_id'] == cluster_id)]['event_id']
                    )]
                    if len(cluster_preds) == 0:
                        continue
                    cluster_signals = extract_signals(
                        cluster_preds,
                        threshold=params['threshold'],
                        min_pending_bars=params['min_pending_bars'],
                        drop_delta=params['drop_delta'],
                        min_pending_peak=params['min_pending_peak'],
                        min_turn_down_bars=params.get('min_turn_down_bars', 1)
                    )
                    if not cluster_signals.empty:
                        tq = compute_trade_quality_metrics(cluster_signals, loader, horizons=[16, 32])
                        trade_quality_by_cluster[int(cluster_id)] = tq
                if trade_quality_by_cluster:
                    artifacts.save_trade_quality_by_cluster(trade_quality_by_cluster)

            enabled_clusters, selectivity_report = evaluate_clusters_selectivity(
                thresholds_by_cluster,
                val_metrics_by_cluster,
                trade_quality_by_cluster=trade_quality_by_cluster,
                min_events=SEL_MIN_EVENTS,
                min_triggered_abs=SEL_MIN_TRIGGERED_ABS,
                min_triggered_frac=SEL_MIN_TRIGGERED_FRAC,
                min_precision_triggered=SEL_MIN_PRECISION_TRIGGERED,
                max_early_far_share=SEL_MAX_EARLY_FAR_SHARE,
                min_inrange_rate=SEL_MIN_INRANGE_RATE,
                max_tail_le_minus10_rate=SEL_MAX_TAIL_LE_MINUS10_RATE,
                min_mfe32_pct_above_2pct=SEL_MIN_MFE32_PCT_ABOVE_2PCT,
                max_mae16_p75=SEL_MAX_MAE16_P75,
                fallback_topk_if_empty=SEL_FALLBACK_TOPK_IF_EMPTY
            )
            thresholds_by_cluster['enabled_clusters'] = enabled_clusters

            artifacts.save_thresholds_by_cluster(thresholds_by_cluster)
            artifacts.save_cluster_selectivity_report(selectivity_report)
            log("INFO", "CLUSTER", f"enabled clusters: {enabled_clusters} / {len(thresholds_by_cluster) - 1}")

            test_metrics_by_cluster = evaluate_by_cluster(test_predictions, thresholds_by_cluster, features_df)
            artifacts.save_metrics_by_cluster(test_metrics_by_cluster, 'test')

            if enabled_clusters:
                val_signals_df = extract_signals_by_cluster(val_predictions, thresholds_by_cluster, enabled_clusters,
                                                            features_df)
                test_signals_df = extract_signals_by_cluster(test_predictions, thresholds_by_cluster, enabled_clusters,
                                                             features_df)
            else:
                log("INFO", "CLUSTER", "enabled_clusters is empty, using global fallback for signal extraction")
                val_signals_df = extract_signals(
                    val_predictions,
                    threshold=global_calibration['threshold'],
                    min_pending_bars=global_calibration['min_pending_bars'],
                    drop_delta=global_calibration['drop_delta'],
                    min_pending_peak=global_calibration['min_pending_peak'],
                    min_turn_down_bars=global_calibration['min_turn_down_bars']
                )
                test_signals_df = extract_signals(
                    test_predictions,
                    threshold=global_calibration['threshold'],
                    min_pending_bars=global_calibration['min_pending_bars'],
                    drop_delta=global_calibration['drop_delta'],
                    min_pending_peak=global_calibration['min_pending_peak'],
                    min_turn_down_bars=global_calibration['min_turn_down_bars']
                )

            artifacts.save_predicted_signals_val(val_signals_df)
            log("INFO", "CLUSTER", f"saved {len(val_signals_df)} VAL predicted signals")

            artifacts.save_predicted_signals_test(test_signals_df)
            log("INFO", "CLUSTER", f"saved {len(test_signals_df)} TEST predicted signals")

            holdout_start = val_end
            holdout_end = test_end
            holdout_output = artifacts.get_path() / "predicted_signals_holdout.csv"
            export_cmd = [
                "python", "-m", "pump_end.cli.export_pump_end_signals",
                "--start-date", holdout_start.strftime('%Y-%m-%d %H:%M:%S'),
                "--end-date", holdout_end.strftime('%Y-%m-%d %H:%M:%S'),
                "--clickhouse-dsn", args.clickhouse_dsn,
                "--model-dir", str(artifacts.get_path()),
                "--output", str(holdout_output),
                "--workers", str(args.build_workers)
            ]
            log("INFO", "CLUSTER", f"running prod-like holdout export for test window [{holdout_start}, {holdout_end})")
            subprocess.run(export_cmd, check=True)
            log("INFO", "CLUSTER", f"saved HOLDOUT predicted signals (prod-like) for test window [{holdout_start}, {holdout_end}) to predicted_signals_holdout.csv")

            pool_signals_df = pd.concat([val_signals_df, test_signals_df], ignore_index=True)
            if 'cluster_id' in pool_signals_df.columns:
                pool_signals_df = pool_signals_df.drop(columns=['cluster_id'])
            pool_signals_df = pool_signals_df.drop_duplicates(subset=['symbol', 'open_time'])
            pool_signals_df = pool_signals_df.sort_values('open_time').reset_index(drop=True)
            artifacts.save_predicted_signals_pool(pool_signals_df)
            log("INFO", "CLUSTER", f"saved {len(pool_signals_df)} POOL predicted signals (val + test)")

            total_val_hit0 = sum(m['event_level']['hit0'] for m in val_metrics_by_cluster.values())
            total_val_events = sum(m['event_level']['n_events'] for m in val_metrics_by_cluster.values())
            total_test_hit0 = sum(m['event_level']['hit0'] for m in test_metrics_by_cluster.values())
            total_test_events = sum(m['event_level']['n_events'] for m in test_metrics_by_cluster.values())

            log("INFO", "CLUSTER", f"val aggregate: hit0={total_val_hit0}/{total_val_events}")
            log("INFO", "CLUSTER", f"test aggregate: hit0={total_test_hit0}/{total_test_events}")

    log("INFO", "TUNE", f"done. artifacts saved to {artifacts.get_path()}")


def main():
    parser = argparse.ArgumentParser(description="Train pump end prediction model")

    parser.add_argument("--mode", type=str, choices=["build-dataset", "tune"], required=True)

    parser.add_argument("--labels", type=str)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clickhouse-dsn", type=str, default=None)
    parser.add_argument("--dataset-parquet", type=str, default=None)

    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)

    parser.add_argument("--build-workers", type=int, default=4)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "build-dataset":
        if not args.labels or not args.clickhouse_dsn:
            parser.error("--labels and --clickhouse-dsn required for build-dataset mode")

    if args.mode == "tune":
        if not args.dataset_parquet:
            parser.error("--dataset-parquet required for tune mode")
        if not args.clickhouse_dsn:
            parser.error("--clickhouse-dsn required for tune mode (needed for prod-like holdout export)")

    artifacts = RunArtifacts(args.out_dir, args.run_name)
    log("INFO", "MAIN", f"run_dir={artifacts.get_path()}")

    if args.mode == "build-dataset":
        run_build_dataset(args, artifacts)
    elif args.mode == "tune":
        run_tune(args, artifacts)


if __name__ == "__main__":
    main()
