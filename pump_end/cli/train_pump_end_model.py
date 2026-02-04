import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from pump_end.features.feature_builder import PumpFeatureBuilder
from pump_end.ml.artifacts import RunArtifacts
from pump_end.ml.dataset import load_labels, build_training_points, deduplicate_points
from pump_end.ml.split import time_split, ratio_split, get_split_info, apply_embargo, clip_points_to_split_bounds
from pump_end.ml.train import train_model, get_feature_columns, get_feature_importance, get_feature_importance_grouped
from pump_end.ml.threshold import threshold_sweep, _prepare_event_data
from pump_end.ml.evaluate import evaluate, evaluate_with_trade_quality
from pump_end.ml.predict import predict_proba, extract_signals
from pump_end.ml.tuning import tune_model, tune_model_both_strategies, train_final_model, get_rule_parameter_grid
from pump_end.ml.feature_schema import prune_feature_columns
from pump_end.ml.clustering import (
    fit_event_clusterer, assign_event_clusters, get_available_cluster_features,
    compute_cluster_feature_summary, compute_cluster_examples, compute_cluster_drift_by_month
)
from pump_end.infra.clickhouse import DataLoader

MAX_EARLY_RATE = 0.35
MAX_MISS_RATE = 0.97


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
        alpha_hit_pre: float,
        beta_early: float,
        gamma_miss: float,
        kappa_early_magnitude: float,
        min_trigger_rate: float,
        max_trigger_rate: float = 1.0,
        grid_from: float = 0.01,
        grid_to: float = 0.30,
        grid_step: float = 0.01
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
            signal_rule=signal_rule,
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
        signal_rule: str,
        alpha_hit1: float,
        alpha_hit_pre: float,
        beta_early: float,
        gamma_miss: float,
        kappa_early_magnitude: float,
        min_trigger_rate: float,
        max_trigger_rate: float,
        artifacts: RunArtifacts,
        cluster_debug_artifacts: bool = False,
        grid_from: float = 0.01,
        grid_to: float = 0.30,
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
        best_sweep_df = None

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
                signal_rule=signal_rule,
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

            if best_sweep_df is None:
                best_sweep_df = sweep_df

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_min_pending_bars = min_pending_bars
                best_drop_delta = drop_delta
                best_min_pending_peak = min_pending_peak
                best_min_turn_down_bars = min_turn_down_bars
                best_sweep_df = sweep_df

        thresholds_by_cluster[int(cluster_id)] = {
            'threshold': best_threshold,
            'min_pending_bars': best_min_pending_bars,
            'drop_delta': best_drop_delta,
            'min_pending_peak': best_min_pending_peak,
            'min_turn_down_bars': best_min_turn_down_bars,
            'n_events': n_events,
            'best_score': best_score
        }

        if cluster_debug_artifacts and best_sweep_df is not None:
            artifacts.save_threshold_sweep_by_cluster(int(cluster_id), best_sweep_df)

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
            signal_rule='pending_turn_down',
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
            signal_rule='pending_turn_down',
            min_pending_bars=params['min_pending_bars'],
            drop_delta=params['drop_delta'],
            min_pending_peak=params['min_pending_peak'],
            min_turn_down_bars=params.get('min_turn_down_bars', 1)
        )

        metrics_by_cluster[int(cluster_id)] = metrics

    return metrics_by_cluster


def prepare_signals_for_backtest(signals_df: pd.DataFrame) -> list[dict]:
    signals_df = signals_df.copy()
    signals_df['open_time'] = pd.to_datetime(signals_df['open_time'])
    signals_df = signals_df.drop_duplicates(subset=['symbol', 'open_time'])
    signals_df = signals_df.sort_values('open_time')

    result = []
    for _, row in signals_df.iterrows():
        result.append({
            'timestamp': row['open_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': row['symbol']
        })
    return result


def run_backtest_optimize(
        signals: list[dict],
        run_name: str,
        base_url: str,
        api_key: str,
        jobs: int,
        timeout_sec: int,
        poll_interval_sec: int,
        artifact_policy: str
) -> dict:
    from backtester.client import submit_experiment, poll_job, get_result

    strategy_grid = {
        "tp_pct": [0.04, 0.05, 0.06, 0.07],
        "sl_pct": [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
        "max_holding_hours": [48, 72],
        "notional_usdt": 1000,
        "fee_pct_per_side": 0.0
    }

    meta = {
        "name": f"{run_name}__bt_opt_val",
        "signals_tf": "15m",
        "artifact_policy": artifact_policy,
        "jobs": jobs
    }

    submit_response = submit_experiment(signals, strategy_grid, meta, base_url, api_key)
    job_id = submit_response['job_id']
    run_id = submit_response['run_id']

    log("INFO", "BACKTEST", f"submitted optimize job_id={job_id} signals={len(signals)}")

    status = poll_job(job_id, base_url, api_key, timeout_sec, poll_interval_sec)

    if status['status'] == 'failed':
        raise RuntimeError(f"Backtest job failed: {status.get('error')}")

    result = get_result(job_id, base_url, api_key)

    return {
        'job_id': job_id,
        'run_id': run_id,
        'result': result
    }


def run_backtest_evaluate(
        signals: list[dict],
        run_name: str,
        tp_pct: float,
        sl_pct: float,
        max_holding_hours: int,
        base_url: str,
        api_key: str,
        jobs: int,
        timeout_sec: int,
        poll_interval_sec: int
) -> dict:
    from backtester.client import submit_experiment, poll_job, get_result

    strategy_grid = {
        "tp_pct": [tp_pct],
        "sl_pct": [sl_pct],
        "max_holding_hours": [max_holding_hours],
        "notional_usdt": 1000,
        "fee_pct_per_side": 0.0
    }

    meta = {
        "name": f"{run_name}__bt_eval_test",
        "signals_tf": "15m",
        "artifact_policy": "best_only",
        "jobs": jobs
    }

    submit_response = submit_experiment(signals, strategy_grid, meta, base_url, api_key)
    job_id = submit_response['job_id']
    run_id = submit_response['run_id']

    log("INFO", "BACKTEST", f"submitted evaluate job_id={job_id} signals={len(signals)}")

    status = poll_job(job_id, base_url, api_key, timeout_sec, poll_interval_sec)

    if status['status'] == 'failed':
        raise RuntimeError(f"Backtest job failed: {status.get('error')}")

    result = get_result(job_id, base_url, api_key)

    return {
        'job_id': job_id,
        'run_id': run_id,
        'result': result
    }


def select_strategy_from_result(
        opt_result: dict,
        base_url: str,
        api_key: str,
        artifact_policy: str,
        min_trades: int = 300,
        max_sl_pct: float = 0.2,
        min_tp_pct: float = 0.04
) -> dict:
    from backtester.client import download_artifact, select_best_strategy_constrained

    if artifact_policy == 'full':
        experiments_csv_path = opt_result['result']['artifacts'].get('experiments_csv')
        if experiments_csv_path:
            csv_content = download_artifact(experiments_csv_path, base_url, api_key)
            return select_best_strategy_constrained(csv_content, min_trades=min_trades, max_sl_pct=max_sl_pct,
                                                    min_tp_pct=min_tp_pct)

    return None


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

    config = vars(args).copy()
    dataset_params = extract_dataset_params(features_df)
    config.update(dataset_params)
    artifacts.save_config(config)

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
        alpha_hit_pre=args.alpha_hit_pre,
        beta_early=args.beta_early,
        gamma_miss=args.gamma_miss,
        kappa_early_magnitude=args.kappa_early_magnitude,
        signal_rule=args.signal_rule,
        min_pending_bars=args.min_pending_bars,
        drop_delta=args.drop_delta,
        min_pending_peak=args.min_pending_peak,
        min_turn_down_bars=args.min_turn_down_bars,
        min_trigger_rate=args.min_trigger_rate,
        max_trigger_rate=args.max_trigger_rate
    )

    artifacts.save_threshold_sweep(sweep_df)
    artifacts.save_best_threshold(best_threshold, {
        'signal_rule': args.signal_rule,
        'min_pending_bars': args.min_pending_bars,
        'drop_delta': args.drop_delta,
        'min_pending_peak': args.min_pending_peak,
        'min_turn_down_bars': args.min_turn_down_bars
    })
    log("INFO", "TRAIN", f"best threshold: {best_threshold:.3f}")

    log("INFO", "TRAIN", "evaluating on val set")
    val_metrics = evaluate(val_predictions, best_threshold, signal_rule=args.signal_rule,
                           min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta,
                           min_pending_peak=args.min_pending_peak, min_turn_down_bars=args.min_turn_down_bars)
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
                            min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta,
                            min_pending_peak=args.min_pending_peak, min_turn_down_bars=args.min_turn_down_bars)
    artifacts.save_metrics(test_metrics, 'test')
    log("INFO", "TRAIN",
        f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

    log("INFO", "TRAIN", "extracting holdout signals")
    signals_df = extract_signals(test_predictions, best_threshold, signal_rule=args.signal_rule,
                                 min_pending_bars=args.min_pending_bars, drop_delta=args.drop_delta,
                                 min_pending_peak=args.min_pending_peak, min_turn_down_bars=args.min_turn_down_bars)
    artifacts.save_predicted_signals(signals_df)
    log("INFO", "TRAIN", f"saved {len(signals_df)} predicted signals to holdout csv")

    log("INFO", "TRAIN", f"done. artifacts saved to {artifacts.get_path()}")


def run_tune(args, artifacts: RunArtifacts):
    log("INFO", "TUNE", f"loading features from {args.dataset_parquet}")
    features_df = pd.read_parquet(args.dataset_parquet)

    config = vars(args).copy()
    dataset_params = extract_dataset_params(features_df)
    config.update(dataset_params)
    artifacts.save_config(config)

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

    if args.tune_strategy == 'both':
        tune_result = tune_model_both_strategies(
            cv_features_df,
            feature_columns,
            time_budget_min=args.time_budget_min,
            fold_months=args.fold_months,
            min_train_months=args.min_train_months,
            signal_rule=args.signal_rule,
            alpha_hit1=args.alpha_hit1,
            alpha_hit_pre=args.alpha_hit_pre,
            beta_early=args.beta_early,
            gamma_miss=args.gamma_miss,
            kappa_early_magnitude=args.kappa_early_magnitude,
            min_trigger_rate=args.min_trigger_rate,
            max_trigger_rate=args.max_trigger_rate,
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
            alpha_hit_pre=args.alpha_hit_pre,
            beta_early=args.beta_early,
            gamma_miss=args.gamma_miss,
            kappa_early_magnitude=args.kappa_early_magnitude,
            min_trigger_rate=args.min_trigger_rate,
            max_trigger_rate=args.max_trigger_rate,
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
            if args.cluster_k > 0:
                log("INFO", "CLUSTER", f"fitting event clusterer with k={args.cluster_k}")

                cluster_features = get_available_cluster_features(features_df)
                log("INFO", "CLUSTER", f"available cluster features: {len(cluster_features)}")

                clusterer, train_event_clusters, cluster_reports = fit_event_clusterer(
                    features_df,
                    train_end,
                    cluster_features=cluster_features,
                    k=args.cluster_k,
                    n_components=args.cluster_n_components,
                    random_state=args.seed
                )

                artifacts.save_cluster_model(clusterer)
                artifacts.save_cluster_quality(cluster_reports)

                if args.cluster_debug_artifacts:
                    artifacts.save_event_clusters(train_event_clusters)

                if cluster_reports.get('cluster_features_dropped'):
                    artifacts.save_cluster_features_dropped(cluster_reports['cluster_features_dropped'])

                cluster_features_used = cluster_reports.get('cluster_features_used', cluster_features)

                cluster_config = {
                    'cluster_features': cluster_features_used,
                    'k': args.cluster_k,
                    'k_used': cluster_reports.get('k_used', args.cluster_k),
                    'n_components': args.cluster_n_components,
                    'train_end': str(train_end),
                    'n_train_events': cluster_reports['n_train_events']
                }
                artifacts.save_cluster_config(cluster_config)

                log("INFO", "CLUSTER",
                    f"cluster quality: silhouette={cluster_reports['silhouette']:.3f} sizes={cluster_reports['cluster_sizes']}")

                features_df = assign_event_clusters(clusterer, features_df, cluster_features_used)

                cluster_feature_summary = compute_cluster_feature_summary(features_df, cluster_features_used)
                artifacts.save_cluster_feature_summary(cluster_feature_summary)

                if args.cluster_debug_artifacts:
                    cluster_examples = compute_cluster_examples(features_df, n_examples=5)
                    artifacts.save_cluster_examples(cluster_examples)

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
                    actual_signal_rule,
                    args.alpha_hit1,
                    args.alpha_hit_pre,
                    args.beta_early,
                    args.gamma_miss,
                    args.kappa_early_magnitude,
                    args.min_trigger_rate,
                    args.max_trigger_rate,
                    artifacts,
                    cluster_debug_artifacts=args.cluster_debug_artifacts,
                    grid_from=args.threshold_grid_from,
                    grid_to=args.threshold_grid_to,
                    grid_step=args.threshold_grid_step
                )

                global_calibration = calibrate_threshold_on_val(
                    final_model,
                    features_df,
                    feature_columns,
                    train_end,
                    val_end,
                    actual_signal_rule,
                    args.alpha_hit1,
                    args.alpha_hit_pre,
                    args.beta_early,
                    args.gamma_miss,
                    args.kappa_early_magnitude,
                    args.min_trigger_rate,
                    args.max_trigger_rate,
                    grid_from=args.threshold_grid_from,
                    grid_to=args.threshold_grid_to,
                    grid_step=args.threshold_grid_step
                )
                log("INFO", "CLUSTER", f"global fallback calibration: threshold={global_calibration['threshold']:.3f}")

                artifacts.save_best_threshold(global_calibration['threshold'], {
                    'signal_rule': actual_signal_rule,
                    'min_pending_bars': global_calibration['min_pending_bars'],
                    'drop_delta': global_calibration['drop_delta'],
                    'min_pending_peak': global_calibration['min_pending_peak'],
                    'min_turn_down_bars': global_calibration['min_turn_down_bars'],
                    'note': 'global_fallback_for_prod_and_empty_clusters'
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
                            signal_rule='pending_turn_down',
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
                    min_events=args.sel_min_events,
                    min_triggered_abs=args.sel_min_triggered_abs,
                    min_triggered_frac=args.sel_min_triggered_frac,
                    min_precision_triggered=args.sel_min_precision_triggered,
                    max_early_far_share=args.sel_max_early_far_share,
                    min_inrange_rate=args.sel_min_inrange_rate,
                    max_tail_le_minus10_rate=args.sel_max_tail_le_minus10_rate,
                    min_mfe32_pct_above_2pct=args.sel_min_mfe32_pct_above_2pct,
                    max_mae16_p75=args.sel_max_mae16_p75,
                    fallback_topk_if_empty=args.sel_fallback_topk_if_empty
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
                        signal_rule=actual_signal_rule,
                        min_pending_bars=global_calibration['min_pending_bars'],
                        drop_delta=global_calibration['drop_delta'],
                        min_pending_peak=global_calibration['min_pending_peak'],
                        min_turn_down_bars=global_calibration['min_turn_down_bars']
                    )
                    test_signals_df = extract_signals(
                        test_predictions,
                        threshold=global_calibration['threshold'],
                        signal_rule=actual_signal_rule,
                        min_pending_bars=global_calibration['min_pending_bars'],
                        drop_delta=global_calibration['drop_delta'],
                        min_pending_peak=global_calibration['min_pending_peak'],
                        min_turn_down_bars=global_calibration['min_turn_down_bars']
                    )

                artifacts.save_predicted_signals_val(val_signals_df)
                log("INFO", "CLUSTER", f"saved {len(val_signals_df)} VAL predicted signals")

                artifacts.save_predicted_signals_test(test_signals_df)
                log("INFO", "CLUSTER", f"saved {len(test_signals_df)} TEST predicted signals")

                if args.cluster_debug_artifacts and enabled_clusters:
                    for cluster_id in enabled_clusters:
                        cluster_signals = val_signals_df[val_signals_df[
                                                             'cluster_id'] == cluster_id] if 'cluster_id' in val_signals_df.columns else pd.DataFrame()
                        if not cluster_signals.empty:
                            artifacts.save_signals_by_cluster(cluster_id, cluster_signals, 'val')
                        cluster_signals = test_signals_df[test_signals_df[
                                                              'cluster_id'] == cluster_id] if 'cluster_id' in test_signals_df.columns else pd.DataFrame()
                        if not cluster_signals.empty:
                            artifacts.save_signals_by_cluster(cluster_id, cluster_signals, 'test')

                holdout_signals_df = test_signals_df.copy()
                if 'cluster_id' in holdout_signals_df.columns:
                    holdout_signals_df = holdout_signals_df.drop(columns=['cluster_id'])
                holdout_signals_df = holdout_signals_df.drop_duplicates(subset=['symbol', 'open_time'])
                holdout_signals_df = holdout_signals_df.sort_values('open_time').reset_index(drop=True)
                artifacts.save_predicted_signals(holdout_signals_df)
                log("INFO", "CLUSTER", f"saved {len(holdout_signals_df)} HOLDOUT predicted signals (= test)")

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

            else:
                log("INFO", "TUNE", f"calibrating threshold on val window [{args.train_end}, {args.val_end})")

                calibration_result = calibrate_threshold_on_val(
                    final_model,
                    features_df,
                    feature_columns,
                    train_end,
                    val_end,
                    actual_signal_rule,
                    args.alpha_hit1,
                    args.alpha_hit_pre,
                    args.beta_early,
                    args.gamma_miss,
                    args.kappa_early_magnitude,
                    args.min_trigger_rate,
                    args.max_trigger_rate,
                    grid_from=args.threshold_grid_from,
                    grid_to=args.threshold_grid_to,
                    grid_step=args.threshold_grid_step
                )

                best_threshold = calibration_result['threshold']
                best_min_pending_bars = calibration_result['min_pending_bars']
                best_drop_delta = calibration_result['drop_delta']
                best_min_pending_peak = calibration_result['min_pending_peak']
                best_min_turn_down_bars = calibration_result['min_turn_down_bars']

                log("INFO", "TUNE",
                    f"calibrated: threshold={best_threshold:.3f} min_pending_bars={best_min_pending_bars} drop_delta={best_drop_delta} min_pending_peak={best_min_pending_peak} min_turn_down_bars={best_min_turn_down_bars}")

                artifacts.save_best_threshold(best_threshold, {
                    'signal_rule': actual_signal_rule,
                    'min_pending_bars': best_min_pending_bars,
                    'drop_delta': best_drop_delta,
                    'min_pending_peak': best_min_pending_peak,
                    'min_turn_down_bars': best_min_turn_down_bars
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
                        min_pending_peak=best_min_pending_peak,
                        min_turn_down_bars=best_min_turn_down_bars,
                        horizons=[16, 32]
                    )
                    log("INFO", "TUNE",
                        f"trade quality score: {test_metrics['trade_quality_score']:.4f}")
                    if 'mfe_short_32' in test_metrics['trade_quality'] and test_metrics['trade_quality'][
                        'mfe_short_32']:
                        mfe_stats = test_metrics['trade_quality']['mfe_short_32']
                        log("INFO", "TUNE",
                            f"MFE_32: median={mfe_stats.get('median', 0):.4f} pct_above_2pct={mfe_stats.get('pct_above_2pct', 0):.2f}")
                else:
                    test_metrics = evaluate(
                        test_predictions,
                        best_threshold,
                        signal_rule=actual_signal_rule,
                        min_pending_bars=best_min_pending_bars,
                        drop_delta=best_drop_delta,
                        min_pending_peak=best_min_pending_peak,
                        min_turn_down_bars=best_min_turn_down_bars
                    )

                artifacts.save_metrics(test_metrics, 'test')
                log("INFO", "TUNE",
                    f"test metrics: hit0={test_metrics['event_level']['hit0_rate']:.3f} early={test_metrics['event_level']['early_rate']:.3f} miss={test_metrics['event_level']['miss_rate']:.3f}")

                val_signals_df = extract_signals(
                    val_predictions,
                    best_threshold,
                    signal_rule=actual_signal_rule,
                    min_pending_bars=best_min_pending_bars,
                    drop_delta=best_drop_delta,
                    min_pending_peak=best_min_pending_peak,
                    min_turn_down_bars=best_min_turn_down_bars
                )
                artifacts.save_predicted_signals_val(val_signals_df)
                log("INFO", "TUNE", f"saved {len(val_signals_df)} VAL predicted signals")

                test_signals_df = extract_signals(
                    test_predictions,
                    best_threshold,
                    signal_rule=actual_signal_rule,
                    min_pending_bars=best_min_pending_bars,
                    drop_delta=best_drop_delta,
                    min_pending_peak=best_min_pending_peak,
                    min_turn_down_bars=best_min_turn_down_bars
                )
                artifacts.save_predicted_signals_test(test_signals_df)
                log("INFO", "TUNE", f"saved {len(test_signals_df)} TEST predicted signals")

                holdout_signals_df = test_signals_df.copy()
                holdout_signals_df = holdout_signals_df.drop_duplicates(subset=['symbol', 'open_time'])
                holdout_signals_df = holdout_signals_df.sort_values('open_time').reset_index(drop=True)
                artifacts.save_predicted_signals(holdout_signals_df)
                log("INFO", "TUNE", f"saved {len(holdout_signals_df)} HOLDOUT predicted signals (= test)")

                pool_signals_df = pd.concat([val_signals_df, test_signals_df], ignore_index=True)
                pool_signals_df = pool_signals_df.drop_duplicates(subset=['symbol', 'open_time'])
                pool_signals_df = pool_signals_df.sort_values('open_time').reset_index(drop=True)
                artifacts.save_predicted_signals_pool(pool_signals_df)
                log("INFO", "TUNE", f"saved {len(pool_signals_df)} POOL predicted signals (val + test)")

            backtest_url = args.backtest_url
            backtest_api_key = args.backtest_api_key

            if backtest_url and backtest_api_key and len(val_signals_df) > 0 and len(test_signals_df) > 0:
                log("INFO", "BACKTEST", f"starting backtest integration")

                run_name = artifacts.get_path().name

                val_signals = prepare_signals_for_backtest(val_signals_df)
                test_signals = prepare_signals_for_backtest(test_signals_df)

                log("INFO", "BACKTEST", f"val_signals={len(val_signals)} test_signals={len(test_signals)}")

                opt_result = run_backtest_optimize(
                    signals=val_signals,
                    run_name=run_name,
                    base_url=backtest_url,
                    api_key=backtest_api_key,
                    jobs=args.backtest_jobs,
                    timeout_sec=args.backtest_timeout_sec,
                    poll_interval_sec=args.backtest_poll_interval_sec,
                    artifact_policy=args.backtest_artifact_policy
                )

                artifacts.save_backtest_opt_val(opt_result)

                selected_strategy = select_strategy_from_result(
                    opt_result,
                    backtest_url,
                    backtest_api_key,
                    args.backtest_artifact_policy,
                    min_trades=args.backtest_min_trades,
                    max_sl_pct=args.backtest_max_sl_pct,
                    min_tp_pct=args.backtest_min_tp_pct
                )

                if selected_strategy is None:
                    log("WARN", "BACKTEST", "no strategy satisfies constraints, skipping test evaluation")

                    backtest_summary = {
                        'val_optimization': {
                            'job_id': opt_result['job_id'],
                            'run_id': opt_result['run_id'],
                            'signals_count': len(val_signals),
                            'selected_strategy': None,
                            'status': 'no_strategy_satisfies_constraints'
                        },
                        'test_evaluation': None
                    }

                    artifacts.save_backtest_summary(backtest_summary)
                else:
                    log("INFO", "BACKTEST",
                        f"selected strategy: tp={selected_strategy['tp_pct']} sl={selected_strategy['sl_pct']} "
                        f"holding={selected_strategy['max_holding_hours']}h winrate={selected_strategy['winrate_all_pct']:.2f}%")

                    eval_result = run_backtest_evaluate(
                        signals=test_signals,
                        run_name=run_name,
                        tp_pct=selected_strategy['tp_pct'],
                        sl_pct=selected_strategy['sl_pct'],
                        max_holding_hours=selected_strategy['max_holding_hours'],
                        base_url=backtest_url,
                        api_key=backtest_api_key,
                        jobs=args.backtest_jobs,
                        timeout_sec=args.backtest_timeout_sec,
                        poll_interval_sec=args.backtest_poll_interval_sec
                    )

                    artifacts.save_backtest_eval_test(eval_result)

                    eval_best = eval_result['result']['best_strategy']['best']

                    backtest_summary = {
                        'val_optimization': {
                            'job_id': opt_result['job_id'],
                            'run_id': opt_result['run_id'],
                            'signals_count': len(val_signals),
                            'selected_strategy': selected_strategy
                        },
                        'test_evaluation': {
                            'job_id': eval_result['job_id'],
                            'run_id': eval_result['run_id'],
                            'signals_count': len(test_signals),
                            'total_trades': eval_best['total_trades'],
                            'winrate_all_pct': eval_best['winrate_all_pct'],
                            'total_pnl_usdt': eval_best['total_pnl_usdt'],
                            'profit_factor': eval_best['profit_factor'],
                            'max_dd_pct': eval_best['max_dd_pct'],
                            'worst_trade_usdt': eval_best['worst_trade_usdt'],
                            'timeout_pct': eval_best['timeout_pct']
                        }
                    }

                    artifacts.save_backtest_summary(backtest_summary)

                    log("INFO", "BACKTEST",
                        f"test results: trades={eval_best['total_trades']} winrate={eval_best['winrate_all_pct']:.2f}% "
                        f"pnl={eval_best['total_pnl_usdt']:.2f} pf={eval_best['profit_factor']:.2f}")

            elif backtest_url and backtest_api_key:
                log("WARN", "BACKTEST", "skipping backtest: not enough signals")

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
    parser.add_argument("--threshold-grid-to", type=float, default=0.95)
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)
    parser.add_argument("--alpha-hit1", type=float, default=0.5)
    parser.add_argument("--alpha-hit-pre", type=float, default=0.25)
    parser.add_argument("--beta-early", type=float, default=2.0)
    parser.add_argument("--gamma-miss", type=float, default=1.0)
    parser.add_argument("--kappa-early-magnitude", type=float, default=0.22)
    parser.add_argument("--min-trigger-rate", type=float, default=0.05)
    parser.add_argument("--max-trigger-rate", type=float, default=0.12)

    parser.add_argument("--signal-rule", type=str, choices=["first_cross", "pending_turn_down", "argmax_per_event"],
                        default="pending_turn_down")
    parser.add_argument("--min-pending-bars", type=int, default=1)
    parser.add_argument("--drop-delta", type=float, default=0.0)
    parser.add_argument("--min-pending-peak", type=float, default=0.0)
    parser.add_argument("--min-turn-down-bars", type=int, default=1)

    parser.add_argument("--tune-strategy", type=str, choices=["threshold", "ranking", "both"],
                        default="threshold")
    parser.add_argument("--time-budget-min", type=int, default=60)
    parser.add_argument("--fold-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=3)

    parser.add_argument("--prune-features", action="store_true", default=False)

    parser.add_argument("--cluster-k", type=int, default=0)
    parser.add_argument("--cluster-n-components", type=int, default=5)
    parser.add_argument("--cluster-debug-artifacts", action="store_true", default=False)

    parser.add_argument("--sel-min-events", type=int, default=50)
    parser.add_argument("--sel-min-triggered-abs", type=int, default=10)
    parser.add_argument("--sel-min-triggered-frac", type=float, default=0.005)
    parser.add_argument("--sel-min-precision-triggered", type=float, default=0.20)
    parser.add_argument("--sel-max-early-far-share", type=float, default=0.65)
    parser.add_argument("--sel-min-inrange-rate", type=float, default=0.55)
    parser.add_argument("--sel-max-tail-le-minus10-rate", type=float, default=0.03)
    parser.add_argument("--sel-min-mfe32-pct-above-2pct", type=float, default=0.75)
    parser.add_argument("--sel-max-mae16-p75", type=float, default=0.40)
    parser.add_argument("--sel-fallback-topk-if-empty", type=int, default=1)

    parser.add_argument("--backtest-url", type=str, default=None)
    parser.add_argument("--backtest-api-key", type=str, default=None)
    parser.add_argument("--backtest-jobs", type=int, default=8)
    parser.add_argument("--backtest-timeout-sec", type=int, default=600)
    parser.add_argument("--backtest-poll-interval-sec", type=int, default=1)
    parser.add_argument("--backtest-artifact-policy", type=str, choices=["best_only", "full"], default="full")
    parser.add_argument("--backtest-min-trades", type=int, default=300)
    parser.add_argument("--backtest-max-sl-pct", type=float, default=0.2)
    parser.add_argument("--backtest-min-tp-pct", type=float, default=0.04)

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

    if args.mode == "build-dataset":
        run_build_dataset(args, artifacts)
    elif args.mode == "train":
        run_train_only(args, artifacts)
    elif args.mode == "tune":
        run_tune(args, artifacts)


if __name__ == "__main__":
    main()
