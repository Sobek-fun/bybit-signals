import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_end_threshold.ml.regime_feature_schema import get_regime_feature_columns
from pump_end_threshold.ml.regime_policy import RegimePolicy


def generate_regime_walk_forward_folds(
        dataset_df: pd.DataFrame,
        fold_months: int = 1,
        min_train_months: int = 3,
        fold_days: int = None,
        min_train_days: int = None,
) -> list:
    times = dataset_df['open_time'].sort_values()
    min_time = times.min()
    max_time = times.max()

    if min_train_days is not None:
        train_offset = pd.DateOffset(days=min_train_days)
    else:
        train_offset = pd.DateOffset(months=min_train_months)

    if fold_days is not None:
        val_offset = pd.DateOffset(days=fold_days)
    else:
        val_offset = pd.DateOffset(months=fold_months)

    folds = []
    current_val_start = min_time + train_offset

    while current_val_start < max_time:
        val_end = current_val_start + val_offset

        if val_end > max_time:
            break

        folds.append({
            'train_start': min_time,
            'train_end': current_val_start,
            'val_start': current_val_start,
            'val_end': val_end,
        })

        current_val_start = val_end

    return folds


def get_regime_hyperparameter_grid() -> list:
    param_grid = {
        'depth': [4, 6],
        'learning_rate': [0.01, 0.03, 0.1],
        'l2_leaf_reg': [1.0, 3.0, 10.0],
        'min_data_in_leaf': [5, 10],
    }
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def get_policy_parameter_grid() -> list:
    policy_grid = {
        'pause_on_threshold': [0.60, 0.65, 0.70],
        'resume_threshold': [0.25, 0.30, 0.35],
        'resume_confirm_signals': [2, 3],
    }
    keys = list(policy_grid.keys())
    values = list(policy_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def train_regime_fold(
        dataset_df: pd.DataFrame,
        feature_columns: list,
        target_col: str,
        params: dict,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
) -> tuple:
    train_df = dataset_df[dataset_df['split'] == 'train']
    val_df = dataset_df[dataset_df['split'] == 'val']

    if len(train_df) == 0 or len(val_df) == 0:
        return None, None

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_val = val_df[feature_columns]
    y_val = val_df[target_col]

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
        auto_class_weights='Balanced',
    )

    model.fit(train_pool, eval_set=val_pool)

    return model, val_df


def evaluate_regime_fold(
        model: CatBoostClassifier,
        val_df: pd.DataFrame,
        feature_columns: list,
        policy_params: dict,
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
) -> dict:
    if model is None or val_df is None or len(val_df) == 0:
        return {'score': -np.inf, 'valid': False}

    X_val = val_df[feature_columns]
    p_bad = model.predict_proba(X_val)[:, 1]
    val_scored = val_df.copy()
    val_scored['p_bad'] = p_bad

    policy = RegimePolicy(
        pause_on_threshold=policy_params['pause_on_threshold'],
        resume_threshold=policy_params['resume_threshold'],
        resume_confirm_signals=policy_params['resume_confirm_signals'],
    )

    result = policy.simulate_pnl(val_scored, p_bad_col='p_bad')

    blocked_share = result.get('blocked_share', 0)
    signals_before = result.get('signals_before', 0)
    signals_after = result.get('signals_after', 0)
    signal_keep_rate = signals_after / max(1, signals_before)

    if blocked_share > max_blocked_share or signal_keep_rate < min_signal_keep_rate:
        return {
            'score': -np.inf,
            'valid': False,
            **result,
            'blocked_share': blocked_share,
            'signal_keep_rate': signal_keep_rate,
        }

    pnl_after = result.get('pnl_after', 0)
    pnl_before = result.get('pnl_before', 0)
    pnl_improvement = pnl_after - pnl_before

    sl_blocked = result.get('sl_blocked', 0)
    tp_blocked = result.get('tp_blocked', 0)
    block_value = 10.0 * sl_blocked - 4.5 * tp_blocked

    tp_block_share_total = tp_blocked / max(1, result.get('tp_kept', 0) + tp_blocked)

    score = pnl_improvement
    if tp_block_share_total > 0.30:
        score -= (tp_block_share_total - 0.30) * 50
    if blocked_share > 0.25:
        score -= (blocked_share - 0.25) * 100

    return {
        'score': score,
        'valid': True,
        'pnl_before': pnl_before,
        'pnl_after': pnl_after,
        'pnl_improvement': pnl_improvement,
        'block_value': block_value,
        **result,
        'signal_keep_rate': signal_keep_rate,
    }


def apply_regime_fold_split(
        dataset_df: pd.DataFrame,
        fold: dict,
        embargo_signals: int = 0
) -> pd.DataFrame:
    df = dataset_df.copy()
    df['split'] = None

    df.loc[
        (df['open_time'] >= fold['train_start']) &
        (df['open_time'] < fold['train_end']),
        'split'
    ] = 'train'

    df.loc[
        (df['open_time'] >= fold['val_start']) &
        (df['open_time'] < fold['val_end']),
        'split'
    ] = 'val'

    if embargo_signals > 0:
        train_df = df[df['split'] == 'train']
        if len(train_df) > embargo_signals:
            last_train_signals = train_df.tail(embargo_signals)
            df.loc[last_train_signals.index, 'split'] = None

    return df[df['split'].notna()].reset_index(drop=True)


def run_regime_cv(
        dataset_df: pd.DataFrame,
        feature_columns: list,
        target_col: str,
        folds: list,
        model_params: dict,
        policy_params: dict,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
        embargo_signals: int = 5,
) -> dict:
    fold_results = []

    for fold_idx, fold in enumerate(folds):
        fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals)
        model, val_df = train_regime_fold(
            fold_df, feature_columns, target_col, model_params,
            iterations=iterations, early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )

        fold_eval = evaluate_regime_fold(
            model, val_df, feature_columns, policy_params,
            max_blocked_share=max_blocked_share,
            min_signal_keep_rate=min_signal_keep_rate,
        )
        fold_eval['fold_idx'] = fold_idx
        fold_results.append(fold_eval)

    valid_scores = [r['score'] for r in fold_results if r.get('valid', False)]

    if not valid_scores:
        return {
            'mean_score': -np.inf,
            'std_score': np.inf,
            'fold_results': fold_results,
            'n_valid_folds': 0,
        }

    return {
        'mean_score': np.mean(valid_scores),
        'std_score': np.std(valid_scores),
        'fold_results': fold_results,
        'n_valid_folds': len(valid_scores),
    }


def tune_regime_guard(
        dataset_df: pd.DataFrame,
        target_col: str = 'target_bad_next_5',
        time_budget_min: int = 60,
        fold_months: int = 1,
        min_train_months: int = 3,
        fold_days: int = None,
        min_train_days: int = None,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
        embargo_signals: int = 5,
) -> dict:
    start_time = time.time()
    time_budget_sec = time_budget_min * 60

    feature_columns = get_regime_feature_columns(dataset_df)

    folds = generate_regime_walk_forward_folds(
        dataset_df, fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
    )

    if not folds:
        raise ValueError("Not enough data to generate walk-forward folds")

    model_grid = get_regime_hyperparameter_grid()
    policy_grid = get_policy_parameter_grid()

    leaderboard = []
    best_score = -np.inf
    best_model_params = None
    best_policy_params = None
    best_cv_result = None

    for model_params in model_grid:
        elapsed = time.time() - start_time
        if elapsed >= time_budget_sec:
            break

        for policy_params in policy_grid:
            elapsed = time.time() - start_time
            if elapsed >= time_budget_sec:
                break

            cv_result = run_regime_cv(
                dataset_df, feature_columns, target_col, folds,
                model_params, policy_params,
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds,
                seed=seed,
                max_blocked_share=max_blocked_share,
                min_signal_keep_rate=min_signal_keep_rate,
                embargo_signals=embargo_signals,
            )

            trial_record = {
                **model_params,
                **{f'policy_{k}': v for k, v in policy_params.items()},
                'mean_score': cv_result['mean_score'],
                'std_score': cv_result['std_score'],
                'n_valid_folds': cv_result['n_valid_folds'],
                'elapsed_sec': time.time() - start_time,
            }
            leaderboard.append(trial_record)

            if cv_result['mean_score'] > best_score:
                best_score = cv_result['mean_score']
                best_model_params = model_params
                best_policy_params = policy_params
                best_cv_result = cv_result

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values('mean_score', ascending=False).reset_index(drop=True)

    return {
        'best_model_params': best_model_params,
        'best_policy_params': best_policy_params,
        'best_score': best_score,
        'best_cv_result': best_cv_result,
        'leaderboard': leaderboard_df,
        'folds': folds,
        'feature_columns': feature_columns,
        'trials_completed': len(leaderboard),
        'time_elapsed_sec': time.time() - start_time,
    }


def train_final_regime_model(
        dataset_df: pd.DataFrame,
        feature_columns: list,
        target_col: str,
        params: dict,
        train_end: datetime,
        iterations: int = 1000,
        seed: int = 42,
) -> CatBoostClassifier:
    train_df = dataset_df[dataset_df['open_time'] < train_end]

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
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
        auto_class_weights='Balanced',
    )

    model.fit(train_pool)
    return model
