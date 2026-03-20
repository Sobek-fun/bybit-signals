import time
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_end_threshold.infra.clickhouse import DataLoader
from pump_end_threshold.ml.evaluate import attach_signal_quality_columns, compute_signal_quality_metrics
from pump_end_threshold.ml.predict import predict_proba, extract_signals_verbose
from pump_end_threshold.ml.threshold import threshold_sweep, _prepare_event_data


def generate_walk_forward_folds(
        points_df: pd.DataFrame,
        fold_months: int = 1,
        min_train_months: int = 3
) -> list:
    event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    event_times = event_times.sort_values('open_time')

    min_time = event_times['open_time'].min()
    max_time = event_times['open_time'].max()

    folds = []
    train_start = min_time

    current_val_start = min_time + pd.DateOffset(months=min_train_months)

    while current_val_start < max_time:
        val_end = current_val_start + pd.DateOffset(months=fold_months)

        if val_end > max_time:
            break

        folds.append({
            'train_start': train_start,
            'train_end': current_val_start,
            'val_start': current_val_start,
            'val_end': val_end
        })

        current_val_start = val_end

    return folds


def apply_fold_split(points_df: pd.DataFrame, fold: dict) -> pd.DataFrame:
    event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')

    train_events = event_times[
        (event_times['open_time'] >= fold['train_start']) &
        (event_times['open_time'] < fold['train_end'])
        ]['event_id']

    val_events = event_times[
        (event_times['open_time'] >= fold['val_start']) &
        (event_times['open_time'] < fold['val_end'])
        ]['event_id']

    points_df = points_df.copy()
    points_df['split'] = None
    points_df.loc[points_df['event_id'].isin(train_events), 'split'] = 'train'
    points_df.loc[points_df['event_id'].isin(val_events), 'split'] = 'val'

    points_df = points_df[points_df['split'].notna()].reset_index(drop=True)

    return points_df


def clip_fold_points(points_df: pd.DataFrame, fold: dict, label_lookahead_bars: int = 0) -> pd.DataFrame:
    points_df = points_df.copy()

    train_mask = points_df['split'] == 'train'
    points_df = points_df[~(train_mask & (points_df['open_time'] >= fold['train_end']))]

    val_mask = points_df['split'] == 'val'
    points_df = points_df[
        ~(val_mask & ((points_df['open_time'] < fold['val_start']) | (points_df['open_time'] >= fold['val_end'])))]

    if label_lookahead_bars > 0:
        lookahead_delta = timedelta(minutes=label_lookahead_bars * 15)
        train_cutoff = fold['train_end'] - lookahead_delta
        val_cutoff = fold['val_end'] - lookahead_delta
        event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time', 'split']].drop_duplicates('event_id')
        train_tail_events = event_times[
            (event_times['split'] == 'train') & (event_times['open_time'] >= train_cutoff)
        ]['event_id']
        val_tail_events = event_times[
            (event_times['split'] == 'val') & (event_times['open_time'] >= val_cutoff)
        ]['event_id']
        tail_events = pd.Index(train_tail_events).union(pd.Index(val_tail_events))
        points_df = points_df[~points_df['event_id'].isin(tail_events)]

    return points_df.reset_index(drop=True)


def apply_fold_embargo(points_df: pd.DataFrame, fold: dict, embargo_bars: int) -> pd.DataFrame:
    if embargo_bars <= 0:
        return points_df

    embargo_delta = timedelta(minutes=embargo_bars * 15)

    event_times = points_df[points_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')

    train_embargo_start = fold['train_end'] - embargo_delta
    train_embargo_end = fold['train_end'] + embargo_delta

    in_embargo = (event_times['open_time'] >= train_embargo_start) & (event_times['open_time'] < train_embargo_end)
    events_to_remove = event_times[in_embargo]['event_id']

    points_df = points_df[~points_df['event_id'].isin(events_to_remove)].copy()

    return points_df


def get_hyperparameter_grid() -> list:
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.1],
        'l2_leaf_reg': [1.0, 3.0, 10.0],
        'min_data_in_leaf': [1, 5, 10]
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def get_rule_parameter_grid() -> list:
    rule_grid = {
        'min_pending_bars': [2, 3, 4],
        'drop_delta': [0.0, 0.01, 0.02, 0.03]
    }

    keys = list(rule_grid.keys())
    values = list(rule_grid.values())

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def train_fold(
        features_df: pd.DataFrame,
        feature_columns: list,
        params: dict,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42
) -> tuple:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    if len(train_df) == 0 or len(val_df) == 0:
        return None, None

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

    w_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    w_val = val_df['sample_weight'].values if 'sample_weight' in val_df.columns else None

    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val, weight=w_val)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        min_data_in_leaf=params['min_data_in_leaf'],
        early_stopping_rounds=early_stopping_rounds,
        random_seed=seed,
        verbose=0,
        eval_metric='Logloss',
        use_best_model=True,
        auto_class_weights='Balanced'
    )

    model.fit(train_pool, eval_set=val_pool)

    return model, val_df


def evaluate_fold(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list,
        signal_rule: str,
        alpha_hit1: float,
        beta_early: float,
        gamma_miss: float,
        threshold_grid_from: float = 0.01,
        threshold_grid_to: float = 0.30,
        threshold_grid_step: float = 0.01,
        early_penalty_threshold: int = -5,
        delta_fp_b: float = 3.0,
        abstain_margin: float = 0.0,
        clickhouse_dsn: str | None = None,
        cv_selection_mode: str = "event_score",
        quality_top_k: int = 8,
) -> dict:
    val_df = features_df[features_df['split'] == 'val']

    if len(val_df) == 0:
        return {'score': -np.inf}

    predictions = predict_proba(model, val_df, feature_columns)
    event_data = _prepare_event_data(predictions)
    quality_mode = cv_selection_mode == "quality_score" and clickhouse_dsn is not None
    loader = DataLoader(clickhouse_dsn) if quality_mode else None

    rule_combinations = get_rule_parameter_grid()

    best_score = -np.inf
    best_threshold = None
    best_min_pending_bars = None
    best_drop_delta = None
    best_metrics = None

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
        score = float(best_row['score'])
        row_for_metrics = best_row
        if loader is not None:
            candidates_df = sweep_df[sweep_df['signal_count'] > 0].copy()
            if not candidates_df.empty:
                candidates_df['prefilter_score'] = candidates_df['score'] + 0.002 * candidates_df['signal_count']
                candidates_df = candidates_df.sort_values('prefilter_score', ascending=False)
                if quality_top_k > 0:
                    candidates_df = candidates_df.head(quality_top_k)
                candidate_signals: dict[float, pd.DataFrame] = {}
                for _, candidate in candidates_df.iterrows():
                    candidate_threshold = float(candidate['threshold'])
                    signals_df = extract_signals_verbose(
                        predictions,
                        candidate_threshold,
                        signal_rule=signal_rule,
                        min_pending_bars=min_pending_bars,
                        drop_delta=drop_delta,
                        abstain_margin=abstain_margin,
                    )
                    if signals_df.empty:
                        continue
                    candidate_signals[candidate_threshold] = signals_df.copy()
                if candidate_signals:
                    sample_df = next(iter(candidate_signals.values()))
                    key_columns = [
                        c for c in ['event_id', 'symbol', 'open_time', 'event_type', 'signal_offset']
                        if c in sample_df.columns
                    ]
                    if not key_columns:
                        key_columns = ['symbol', 'open_time']
                    combined_signals = pd.concat(list(candidate_signals.values()), ignore_index=True)
                    combined_signals = combined_signals.drop_duplicates(subset=key_columns)
                    combined_quality = attach_signal_quality_columns(combined_signals, loader, horizons=[32])
                    combined_quality['__signal_key'] = (
                        combined_quality[key_columns].astype(str).agg('|'.join, axis=1)
                    )
                    combined_quality = combined_quality.drop_duplicates(subset='__signal_key')
                    best_quality_score = -np.inf
                    best_quality_row = None
                    for candidate_threshold, signals_df in candidate_signals.items():
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
                            pullback_threshold=0.03,
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
                        quality_score = (
                            5.0 * clean_2_3_share_h32
                            + 3.0 * clean_retrace_precision_2_3_h32
                            + 2.0 * pullback_before_squeeze_share_2_3_h32
                            + 20.0 * net_edge_median_h32
                            + 8.0 * pullback_median_h32
                            - 18.0 * squeeze_p75_h32
                            - 5.0 * dirty_no_pullback_2_3_share_h32
                            + 0.01 * min(signal_count_quality, 120)
                            - 0.18 * max(0, 30 - signal_count_quality)
                        )
                        if quality_score > best_quality_score:
                            best_quality_score = quality_score
                            best_quality_row = sweep_df[sweep_df['threshold'] == candidate_threshold].iloc[0]
                    if best_quality_row is not None:
                        score = float(best_quality_score)
                        row_for_metrics = best_quality_row

        metrics = {
            'hit0_rate': row_for_metrics['hit0_rate'],
            'hit1_rate': row_for_metrics['hit1_rate'],
            'early_rate': row_for_metrics['early_rate'],
            'late_rate': row_for_metrics['late_rate'],
            'miss_rate': row_for_metrics['miss_rate'],
            'fp_b_rate': row_for_metrics.get('fp_b_rate', 0),
            'signal_count': row_for_metrics.get('signal_count', 0),
            'median_pred_offset': row_for_metrics['median_offset'],
            'n_events': row_for_metrics['n_events']
        }

        if loader is None:
            median_offset = metrics.get('median_pred_offset')
            if median_offset is not None and median_offset < early_penalty_threshold:
                early_penalty = abs(median_offset - early_penalty_threshold) * 0.1
            else:
                early_penalty = 0
            score = score - early_penalty

        if score > best_score:
            best_score = score
            best_threshold = float(row_for_metrics['threshold'])
            best_min_pending_bars = min_pending_bars
            best_drop_delta = drop_delta
            best_metrics = metrics

    return {
        'score': best_score,
        'threshold': best_threshold,
        'min_pending_bars': best_min_pending_bars,
        'drop_delta': best_drop_delta,
        'hit0_rate': best_metrics['hit0_rate'] if best_metrics else 0,
        'hit1_rate': best_metrics['hit1_rate'] if best_metrics else 0,
        'early_rate': best_metrics['early_rate'] if best_metrics else 0,
        'late_rate': best_metrics['late_rate'] if best_metrics else 0,
        'miss_rate': best_metrics['miss_rate'] if best_metrics else 0,
        'fp_b_rate': best_metrics['fp_b_rate'] if best_metrics else 0,
        'signal_count': best_metrics['signal_count'] if best_metrics else 0,
        'median_pred_offset': best_metrics.get('median_pred_offset') if best_metrics else None,
        'n_events': best_metrics['n_events'] if best_metrics else 0
    }


def run_cv(
        features_df: pd.DataFrame,
        feature_columns: list,
        folds: list,
        params: dict,
        signal_rule: str = 'pending_turn_down',
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0,
        threshold_grid_from: float = 0.01,
        threshold_grid_to: float = 0.30,
        threshold_grid_step: float = 0.01,
        delta_fp_b: float = 3.0,
        abstain_margin: float = 0.0,
        embargo_bars: int = 0,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        tune_strategy: str = 'threshold',
        clickhouse_dsn: str | None = None,
        cv_selection_mode: str = "event_score",
        quality_top_k: int = 8,
        label_lookahead_bars: int = 0,
) -> dict:
    fold_results = []

    actual_signal_rule = signal_rule

    for fold_idx, fold in enumerate(folds):
        fold_df = apply_fold_split(features_df, fold)
        fold_df = apply_fold_embargo(fold_df, fold, embargo_bars)
        fold_df = clip_fold_points(fold_df, fold, label_lookahead_bars=label_lookahead_bars)

        model, _ = train_fold(
            fold_df,
            feature_columns,
            params,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed
        )

        if model is None:
            continue

        fold_metrics = evaluate_fold(
            model,
            fold_df,
            feature_columns,
            actual_signal_rule,
            alpha_hit1,
            beta_early,
            gamma_miss,
            threshold_grid_from=threshold_grid_from,
            threshold_grid_to=threshold_grid_to,
            threshold_grid_step=threshold_grid_step,
            delta_fp_b=delta_fp_b,
            abstain_margin=abstain_margin,
            clickhouse_dsn=clickhouse_dsn,
            cv_selection_mode=cv_selection_mode,
            quality_top_k=quality_top_k,
        )

        fold_metrics['fold_idx'] = fold_idx
        fold_results.append(fold_metrics)

    if not fold_results:
        return {
            'mean_score': -np.inf,
            'std_score': np.inf,
            'mean_threshold': None,
            'fold_results': []
        }

    scores = [r['score'] for r in fold_results]
    thresholds = [r['threshold'] for r in fold_results if r['threshold'] is not None]

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_threshold': np.mean(thresholds) if thresholds else None,
        'fold_results': fold_results
    }


def tune_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        time_budget_min: int = 60,
        fold_months: int = 1,
        min_train_months: int = 3,
        signal_rule: str = 'pending_turn_down',
        alpha_hit1: float = 0.5,
        beta_early: float = 2.0,
        gamma_miss: float = 1.0,
        threshold_grid_from: float = 0.01,
        threshold_grid_to: float = 0.30,
        threshold_grid_step: float = 0.01,
        delta_fp_b: float = 3.0,
        abstain_margin: float = 0.0,
        embargo_bars: int = 0,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        tune_strategy: str = 'threshold',
        clickhouse_dsn: str | None = None,
        cv_selection_mode: str = "event_score",
        quality_top_k: int = 8,
        label_lookahead_bars: int = 0,
) -> dict:
    start_time = time.time()
    time_budget_sec = time_budget_min * 60

    folds = generate_walk_forward_folds(features_df, fold_months, min_train_months)

    if len(folds) == 0:
        raise ValueError("Not enough data to generate walk-forward folds")

    param_combinations = get_hyperparameter_grid()

    leaderboard = []
    best_result = None
    best_params = None
    best_score = -np.inf

    for trial_idx, params in enumerate(param_combinations):
        elapsed = time.time() - start_time
        if elapsed >= time_budget_sec:
            break

        cv_result = run_cv(
            features_df,
            feature_columns,
            folds,
            params,
            signal_rule=signal_rule,
            alpha_hit1=alpha_hit1,
            beta_early=beta_early,
            gamma_miss=gamma_miss,
            threshold_grid_from=threshold_grid_from,
            threshold_grid_to=threshold_grid_to,
            threshold_grid_step=threshold_grid_step,
            delta_fp_b=delta_fp_b,
            abstain_margin=abstain_margin,
            embargo_bars=embargo_bars,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            tune_strategy=tune_strategy,
            clickhouse_dsn=clickhouse_dsn,
            cv_selection_mode=cv_selection_mode,
            quality_top_k=quality_top_k,
            label_lookahead_bars=label_lookahead_bars,
        )

        fold_results = cv_result.get('fold_results', [])
        mean_hit0_rate = np.mean([r.get('hit0_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_hit1_rate = np.mean([r.get('hit1_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_early_rate = np.mean([r.get('early_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_late_rate = np.mean([r.get('late_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_miss_rate = np.mean([r.get('miss_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_fp_b_rate = np.mean([r.get('fp_b_rate', 0) for r in fold_results]) if fold_results else np.nan
        mean_signal_count = np.mean([r.get('signal_count', 0) for r in fold_results]) if fold_results else np.nan
        median_offsets = [r.get('median_pred_offset') for r in fold_results if r.get('median_pred_offset') is not None]
        mean_median_pred_offset = np.mean(median_offsets) if median_offsets else np.nan

        trial_record = {
            'trial_idx': trial_idx,
            **params,
            'mean_score': cv_result['mean_score'],
            'std_score': cv_result['std_score'],
            'mean_hit0_rate': mean_hit0_rate,
            'mean_hit1_rate': mean_hit1_rate,
            'mean_early_rate': mean_early_rate,
            'mean_late_rate': mean_late_rate,
            'mean_miss_rate': mean_miss_rate,
            'mean_fp_b_rate': mean_fp_b_rate,
            'mean_signal_count': mean_signal_count,
            'mean_median_pred_offset': mean_median_pred_offset,
            'elapsed_sec': time.time() - start_time
        }
        leaderboard.append(trial_record)

        if cv_result['mean_score'] > best_score:
            best_score = cv_result['mean_score']
            best_params = params
            best_result = cv_result

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values('mean_score', ascending=False).reset_index(drop=True)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_cv_result': best_result,
        'leaderboard': leaderboard_df,
        'folds': folds,
        'trials_completed': len(leaderboard),
        'time_elapsed_sec': time.time() - start_time,
        'tune_strategy': tune_strategy
    }


def train_final_model(
        features_df: pd.DataFrame,
        feature_columns: list,
        params: dict,
        train_end: datetime,
        iterations: int = 1000,
        seed: int = 42,
        tune_strategy: str = 'threshold',
        label_lookahead_bars: int = 0,
) -> CatBoostClassifier:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    train_events = event_times[event_times['open_time'] < train_end]['event_id']
    if label_lookahead_bars > 0:
        lookahead_delta = timedelta(minutes=label_lookahead_bars * 15)
        train_cutoff = train_end - lookahead_delta
        train_events = event_times[event_times['open_time'] < train_cutoff]['event_id']

    train_df = features_df[
        (features_df['event_id'].isin(train_events)) &
        (features_df['open_time'] < train_end)
        ]

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    w_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    train_pool = Pool(X_train, y_train, weight=w_train)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        min_data_in_leaf=params['min_data_in_leaf'],
        random_seed=seed,
        verbose=100,
        eval_metric='Logloss',
        auto_class_weights='Balanced'
    )

    model.fit(train_pool)

    return model
