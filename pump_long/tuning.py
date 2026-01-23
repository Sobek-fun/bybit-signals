import time
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_long.threshold import threshold_sweep_long, _prepare_event_data


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


def clip_fold_points(points_df: pd.DataFrame, fold: dict) -> pd.DataFrame:
    points_df = points_df.copy()

    train_mask = points_df['split'] == 'train'
    points_df = points_df[~(train_mask & (points_df['open_time'] >= fold['train_end']))]

    val_mask = points_df['split'] == 'val'
    points_df = points_df[
        ~(val_mask & ((points_df['open_time'] < fold['val_start']) | (points_df['open_time'] >= fold['val_end'])))]

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


def compute_sample_weights(offsets: np.ndarray, y: np.ndarray, neg_before: int = 60) -> np.ndarray:
    weights = np.ones(len(offsets), dtype=np.float64)

    for i in range(len(offsets)):
        offset = offsets[i]
        label = y[i]

        if label == 1:
            weights[i] = 5.0
        elif offset < 0:
            distance = abs(offset)
            weights[i] = 1.0 + (distance / neg_before) * 2.0
        else:
            weights[i] = 1.5

    return weights


def predict_proba_long(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list
) -> pd.DataFrame:
    X = features_df[feature_columns]
    proba = model.predict_proba(X)[:, 1]

    result_df = features_df[['event_id', 'symbol', 'open_time', 'offset', 'y', 'split']].copy()
    result_df['p_long'] = proba

    return result_df


def train_fold(
        features_df: pd.DataFrame,
        feature_columns: list,
        params: dict,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42,
        use_sample_weights: bool = True,
        neg_before: int = 60
) -> tuple:
    train_df = features_df[features_df['split'] == 'train']
    val_df = features_df[features_df['split'] == 'val']

    if len(train_df) == 0 or len(val_df) == 0:
        return None, None

    X_train = train_df[feature_columns]
    y_train = train_df['y']
    X_val = val_df[feature_columns]
    y_val = val_df['y']

    if use_sample_weights:
        train_weights = compute_sample_weights(
            train_df['offset'].values,
            train_df['y'].values,
            neg_before
        )
        train_pool = Pool(X_train, y_train, weight=train_weights)
    else:
        train_pool = Pool(X_train, y_train)

    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        min_data_in_leaf=params['min_data_in_leaf'],
        early_stopping_rounds=early_stopping_rounds,
        thread_count=thread_count,
        random_seed=seed,
        verbose=0,
        eval_metric='Logloss',
        use_best_model=True,
        auto_class_weights='Balanced'
    )

    model.fit(train_pool, eval_set=val_pool)

    return model, val_df


def evaluate_fold_long(
        model: CatBoostClassifier,
        features_df: pd.DataFrame,
        feature_columns: list,
        signal_rule: str,
        alpha_hitM1: float,
        beta_early: float,
        beta_late: float,
        gamma_miss: float,
        lambda_offset: float = 0.02,
        hysteresis_delta: float = 0.05
) -> dict:
    val_df = features_df[features_df['split'] == 'val']

    if len(val_df) == 0:
        return {'score': -np.inf}

    predictions = predict_proba_long(model, val_df, feature_columns)
    event_data = _prepare_event_data(predictions)

    threshold, sweep_df = threshold_sweep_long(
        predictions,
        alpha_hitM1=alpha_hitM1,
        beta_early=beta_early,
        beta_late=beta_late,
        gamma_miss=gamma_miss,
        lambda_offset=lambda_offset,
        signal_rule=signal_rule,
        hysteresis_delta=hysteresis_delta,
        event_data=event_data
    )

    best_row = sweep_df[sweep_df['threshold'] == threshold].iloc[0]

    return {
        'score': best_row['score'],
        'threshold': threshold,
        'hit0_rate': best_row['hit0_rate'],
        'hitM1_rate': best_row['hitM1_rate'],
        'early_rate': best_row['early_rate'],
        'late_rate': best_row['late_rate'],
        'miss_rate': best_row['miss_rate'],
        'median_offset': best_row['median_offset'],
        'n_events': best_row['n_events']
    }


def run_cv_long(
        features_df: pd.DataFrame,
        feature_columns: list,
        folds: list,
        params: dict,
        signal_rule: str = 'cross_up',
        alpha_hitM1: float = 0.8,
        beta_early: float = 5.0,
        beta_late: float = 3.0,
        gamma_miss: float = 0.3,
        lambda_offset: float = 0.02,
        embargo_bars: int = 0,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42,
        use_sample_weights: bool = True,
        neg_before: int = 60,
        hysteresis_delta: float = 0.05
) -> dict:
    fold_results = []

    for fold_idx, fold in enumerate(folds):
        fold_df = apply_fold_split(features_df, fold)
        fold_df = apply_fold_embargo(fold_df, fold, embargo_bars)
        fold_df = clip_fold_points(fold_df, fold)

        model, _ = train_fold(
            fold_df,
            feature_columns,
            params,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            thread_count=thread_count,
            seed=seed,
            use_sample_weights=use_sample_weights,
            neg_before=neg_before
        )

        if model is None:
            continue

        fold_metrics = evaluate_fold_long(
            model,
            fold_df,
            feature_columns,
            signal_rule,
            alpha_hitM1,
            beta_early,
            beta_late,
            gamma_miss,
            lambda_offset,
            hysteresis_delta
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


def tune_model_long(
        features_df: pd.DataFrame,
        feature_columns: list,
        time_budget_min: int = 60,
        fold_months: int = 1,
        min_train_months: int = 3,
        signal_rule: str = 'cross_up',
        alpha_hitM1: float = 0.8,
        beta_early: float = 5.0,
        beta_late: float = 3.0,
        gamma_miss: float = 0.3,
        lambda_offset: float = 0.02,
        embargo_bars: int = 0,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int = 42,
        use_sample_weights: bool = True,
        neg_before: int = 60,
        hysteresis_delta: float = 0.05
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

        cv_result = run_cv_long(
            features_df,
            feature_columns,
            folds,
            params,
            signal_rule=signal_rule,
            alpha_hitM1=alpha_hitM1,
            beta_early=beta_early,
            beta_late=beta_late,
            gamma_miss=gamma_miss,
            lambda_offset=lambda_offset,
            embargo_bars=embargo_bars,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            thread_count=thread_count,
            seed=seed,
            use_sample_weights=use_sample_weights,
            neg_before=neg_before,
            hysteresis_delta=hysteresis_delta
        )

        trial_record = {
            'trial_idx': trial_idx,
            **params,
            'mean_score': cv_result['mean_score'],
            'std_score': cv_result['std_score'],
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
        'time_elapsed_sec': time.time() - start_time
    }


def train_final_model_long(
        features_df: pd.DataFrame,
        feature_columns: list,
        params: dict,
        train_end: datetime,
        iterations: int = 1000,
        thread_count: int = -1,
        seed: int = 42,
        use_sample_weights: bool = True,
        neg_before: int = 60
) -> CatBoostClassifier:
    event_times = features_df[features_df['offset'] == 0][['event_id', 'open_time']].drop_duplicates('event_id')
    train_events = event_times[event_times['open_time'] < train_end]['event_id']

    train_df = features_df[
        (features_df['event_id'].isin(train_events)) &
        (features_df['open_time'] < train_end)
        ]

    X_train = train_df[feature_columns]
    y_train = train_df['y']

    if use_sample_weights:
        train_weights = compute_sample_weights(
            train_df['offset'].values,
            train_df['y'].values,
            neg_before
        )
        train_pool = Pool(X_train, y_train, weight=train_weights)
    else:
        train_pool = Pool(X_train, y_train)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        min_data_in_leaf=params['min_data_in_leaf'],
        thread_count=thread_count,
        random_seed=seed,
        verbose=100,
        eval_metric='Logloss',
        auto_class_weights='Balanced'
    )

    model.fit(train_pool)

    return model
