import random
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pump_end_threshold.ml.regime_evaluate import evaluate_regime, compute_cv_score
from pump_end_threshold.ml.regime_feature_schema import get_regime_feature_columns
from pump_end_threshold.ml.regime_policy import RegimePolicy


def _safe_binary_metrics(y_true: pd.Series, y_score: np.ndarray) -> tuple:
    y = pd.Series(y_true).dropna()
    if len(y) == 0:
        return None, None, None

    valid_mask = pd.Series(y_true).notna().values
    y_score_valid = y_score[valid_mask]
    y_values = y.values

    brier = float(np.mean((y_score_valid - y_values) ** 2))
    classes = np.unique(y_values)
    if len(classes) < 2:
        return None, None, brier

    from sklearn.metrics import roc_auc_score, average_precision_score
    ap = float(average_precision_score(y_values, y_score_valid))
    auc = float(roc_auc_score(y_values, y_score_valid))
    return ap, auc, brier


def generate_regime_walk_forward_folds(
        dataset_df: pd.DataFrame,
        fold_months: int = 1,
        min_train_months: int = 3,
        fold_days: int = 14,
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
        'depth': [4, 6, 8],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1.0, 3.0, 10.0, 30.0, 100.0],
        'min_data_in_leaf': [1, 5, 10, 25, 50],
        'random_strength': [0.0, 1.0, 3.0],
        'bagging_temperature': [0.0, 0.5, 1.0],
        'rsm': [0.5, 0.75, 1.0],
    }
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def get_probe_policy_grid(preset: str, max_blocked_share: float = 0.35) -> list:
    if preset == 'selective_local':
        return [
            {'pause_on_quantile': 0.88, 'resume_quantile': 0.65, 'resume_confirm_signals': 1},
            {'pause_on_quantile': 0.90, 'resume_quantile': 0.70, 'resume_confirm_signals': 1},
            {'pause_on_quantile': 0.92, 'resume_quantile': 0.75, 'resume_confirm_signals': 1},
        ]
    probe_quantile = 0.88 if max_blocked_share < 0.25 else 0.80
    return [
        {'pause_on_quantile': probe_quantile, 'resume_quantile': 0.55, 'resume_confirm_signals': 2},
    ]


def get_policy_parameter_grid(preset: str = 'default') -> list:
    if preset == 'conservative':
        policy_grid = {
            'pause_on_quantile': [0.85, 0.90, 0.95],
            'resume_quantile': [0.40, 0.50, 0.60],
            'resume_confirm_signals': [2, 3, 4],
        }
    elif preset == 'aggressive':
        policy_grid = {
            'pause_on_quantile': [0.75, 0.80, 0.85],
            'resume_quantile': [0.50, 0.60, 0.70],
            'resume_confirm_signals': [1, 2],
        }
    elif preset == 'selective_local':
        policy_grid = {
            'pause_on_quantile': [0.85, 0.88, 0.90, 0.92, 0.95],
            'resume_quantile': [0.60, 0.65, 0.70, 0.75],
            'resume_confirm_signals': [1],
        }
    elif preset == 'low':
        policy_grid = {
            'pause_on_threshold': [0.48, 0.52, 0.56, 0.60],
            'resume_threshold': [0.28, 0.32, 0.36, 0.40],
            'resume_confirm_signals': [1, 2],
        }
    else:
        policy_grid = {
            'pause_on_quantile': [0.80, 0.85, 0.90],
            'resume_quantile': [0.45, 0.55, 0.65],
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
        iterations: int = 10000,
        early_stopping_rounds: int = 300,
        seed: int = 42,
        auto_class_weights: str = None,
) -> tuple:
    train_df = dataset_df[dataset_df['split'] == 'train']
    val_df = dataset_df[dataset_df['split'] == 'val']

    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])

    if len(train_df) == 0 or len(val_df) == 0:
        return None, None

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_val = val_df[feature_columns]
    y_val = val_df[target_col]

    w_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    w_val = val_df['sample_weight'].values if 'sample_weight' in val_df.columns else None

    if w_train is not None:
        valid_mask = ~np.isnan(w_train)
        if not valid_mask.all():
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            w_train = w_train[valid_mask]

    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val, weight=w_val if w_val is None or not np.any(np.isnan(w_val)) else None)

    catboost_params = {
        'iterations': iterations,
        'depth': params['depth'],
        'learning_rate': params['learning_rate'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'random_strength': params.get('random_strength', 1.0),
        'bagging_temperature': params.get('bagging_temperature', 1.0),
        'rsm': params.get('rsm', 1.0),
        'early_stopping_rounds': early_stopping_rounds,
        'random_seed': seed,
        'verbose': 0,
        'eval_metric': 'Logloss',
        'use_best_model': True,
    }

    if auto_class_weights and w_train is None:
        catboost_params['auto_class_weights'] = auto_class_weights

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=val_pool)

    return model, val_df


def resolve_policy_params(policy_params: dict, p_bad: np.ndarray) -> dict:
    if 'pause_on_quantile' in policy_params:
        return {
            'pause_on_threshold': float(np.quantile(p_bad, policy_params['pause_on_quantile'])),
            'resume_threshold': float(np.quantile(p_bad, policy_params['resume_quantile'])),
            'resume_confirm_signals': policy_params['resume_confirm_signals'],
        }
    return {
        'pause_on_threshold': policy_params['pause_on_threshold'],
        'resume_threshold': policy_params['resume_threshold'],
        'resume_confirm_signals': policy_params['resume_confirm_signals'],
    }


def evaluate_regime_fold(
        model: CatBoostClassifier,
        val_df: pd.DataFrame,
        feature_columns: list,
        policy_params: dict,
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
        score_mode: str = 'pnl_improvement',
        target_col: str = 'target_bad_next_5',
) -> dict:
    if model is None or val_df is None or len(val_df) == 0:
        return {'score': -np.inf, 'valid': False, 'no_op': True}

    X_val = val_df[feature_columns]
    p_bad = model.predict_proba(X_val)[:, 1]
    val_scored = val_df.copy()
    val_scored['p_bad'] = p_bad

    resolved = resolve_policy_params(policy_params, p_bad)

    policy = RegimePolicy(
        pause_on_threshold=resolved['pause_on_threshold'],
        resume_threshold=resolved['resume_threshold'],
        resume_confirm_signals=resolved['resume_confirm_signals'],
    )

    filtered = policy.apply(val_scored, p_bad_col='p_bad')

    metrics = evaluate_regime(filtered, target_col=target_col)

    blocked_share = metrics.get('blocked_share', 0)
    signal_keep_rate = metrics.get('signal_keep_rate', 1.0)

    if blocked_share > max_blocked_share or signal_keep_rate < min_signal_keep_rate:
        return {
            'score': -np.inf,
            'valid': False,
            'resolved_pause_threshold': resolved['pause_on_threshold'],
            'resolved_resume_threshold': resolved['resume_threshold'],
            **metrics,
        }

    score = compute_cv_score(
        metrics, score_mode=score_mode,
        max_blocked_share=max_blocked_share,
        min_signal_keep_rate=min_signal_keep_rate,
    )

    return {
        'score': score,
        'valid': True,
        'no_op': blocked_share == 0,
        'resolved_pause_threshold': resolved['pause_on_threshold'],
        'resolved_resume_threshold': resolved['resume_threshold'],
        **metrics,
    }


def apply_regime_fold_split(
        dataset_df: pd.DataFrame,
        fold: dict,
        embargo_signals: int = 0,
        embargo_hours: float = 0,
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

    if embargo_hours > 0:
        embargo_cutoff = fold['train_end'] - pd.Timedelta(hours=embargo_hours)
        df.loc[
            (df['split'] == 'train') &
            (df['open_time'] >= embargo_cutoff),
            'split'
        ] = None

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
        embargo_hours: float = 0,
        score_mode: str = 'pnl_improvement',
) -> dict:
    fold_results = []

    for fold_idx, fold in enumerate(folds):
        fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals, embargo_hours=embargo_hours)
        model, val_df = train_regime_fold(
            fold_df, feature_columns, target_col, model_params,
            iterations=iterations, early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )

        fold_eval = evaluate_regime_fold(
            model, val_df, feature_columns, policy_params,
            max_blocked_share=max_blocked_share,
            min_signal_keep_rate=min_signal_keep_rate,
            score_mode=score_mode,
            target_col=target_col,
        )
        fold_eval['fold_idx'] = fold_idx
        fold_eval['train_start'] = fold['train_start']
        fold_eval['train_end'] = fold['train_end']
        fold_eval['val_start'] = fold['val_start']
        fold_eval['val_end'] = fold['val_end']
        fold_eval['train_size'] = len(fold_df[fold_df['split'] == 'train'])
        fold_eval['val_size'] = len(fold_df[fold_df['split'] == 'val'])
        fold_results.append(fold_eval)

    valid_scores = [r['score'] for r in fold_results if r.get('valid', False)]
    valid_results = [r for r in fold_results if r.get('valid', False)]

    if not valid_scores:
        return {
            'mean_score': -np.inf,
            'std_score': np.inf,
            'weighted_mean_score': -np.inf,
            'fold_results': fold_results,
            'n_valid_folds': 0,
            'n_no_op_folds': sum(1 for r in fold_results if r.get('no_op', False)),
        }

    weights = [r.get('signals_before', 1) for r in valid_results]
    weighted_mean = np.average(valid_scores, weights=weights) if sum(weights) > 0 else np.mean(valid_scores)

    n_no_op = sum(1 for r in fold_results if r.get('no_op', False))
    no_op_penalty = -500 * n_no_op / len(fold_results) if fold_results else 0

    return {
        'mean_score': np.mean(valid_scores),
        'std_score': np.std(valid_scores),
        'weighted_mean_score': weighted_mean + no_op_penalty,
        'fold_results': fold_results,
        'n_valid_folds': len(valid_scores),
        'n_no_op_folds': n_no_op,
    }


def tune_model_hyperparameters(
        dataset_df: pd.DataFrame,
        target_col: str = 'target_bad_next_5',
        time_budget_min: int = 30,
        fold_months: int = 1,
        min_train_months: int = 3,
        fold_days: int = None,
        min_train_days: int = None,
        iterations: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        embargo_signals: int = 5,
        embargo_hours: float = 0,
        min_valid_folds: int = 2,
        n_seeds: int = 3,
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
        score_mode: str = 'block_value',
        model_selection_mode: str = 'downstream_cv',
        feature_profile: str = None,
        policy_grid_preset: str = 'default',
) -> dict:
    start_time = time.time()
    time_budget_sec = time_budget_min * 60

    feature_columns = get_regime_feature_columns(dataset_df, feature_profile=feature_profile)

    folds = generate_regime_walk_forward_folds(
        dataset_df, fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
    )

    if not folds:
        raise ValueError("Not enough data to generate walk-forward folds")

    model_grid = get_regime_hyperparameter_grid()
    random.shuffle(model_grid)

    best_score = -np.inf
    best_params = None
    results = []
    probe_policies = get_probe_policy_grid(policy_grid_preset, max_blocked_share)

    for params in model_grid:
        elapsed = time.time() - start_time
        if elapsed >= time_budget_sec:
            break

        fold_scores = []
        downstream_scores = []
        for seed_offset in range(n_seeds):
            curr_seed = seed + seed_offset * 123
            for fold_idx, fold in enumerate(folds):
                fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals, embargo_hours=embargo_hours)
                model, val_df = train_regime_fold(
                    fold_df, feature_columns, target_col, params,
                    iterations=iterations, early_stopping_rounds=early_stopping_rounds,
                    seed=curr_seed,
                )

                if model is None or val_df is None or len(val_df) == 0:
                    continue

                X_val = val_df[feature_columns]
                p_bad = model.predict_proba(X_val)[:, 1]

                y_val = val_df[target_col]
                ap, auc, brier = _safe_binary_metrics(y_val, p_bad)
                if ap is None:
                    continue
                fold_scores.append({
                    'ap': ap, 'auc': auc, 'brier': brier,
                    'seed': curr_seed, 'fold': fold_idx,
                })
                for probe_policy in probe_policies:
                    downstream_eval = evaluate_regime_fold(
                        model=model,
                        val_df=val_df,
                        feature_columns=feature_columns,
                        policy_params=probe_policy,
                        max_blocked_share=max_blocked_share,
                        min_signal_keep_rate=min_signal_keep_rate,
                        score_mode=score_mode,
                        target_col=target_col,
                    )
                    downstream_score = downstream_eval.get('score', -np.inf) if downstream_eval.get('valid', False) else -np.inf
                    if np.isfinite(downstream_score):
                        downstream_scores.append(float(downstream_score))

        if len(fold_scores) >= min_valid_folds:
            mean_ap = np.mean([s['ap'] for s in fold_scores])
            mean_auc = np.mean([s['auc'] for s in fold_scores])
            mean_brier = np.mean([s['brier'] for s in fold_scores])
            mean_downstream_cv = np.mean(downstream_scores) if downstream_scores else -np.inf
            model_score = mean_downstream_cv if model_selection_mode == 'downstream_cv' else mean_ap

            if model_score > best_score:
                best_score = model_score
                best_params = params

            results.append({
                **params,
                'mean_ap': mean_ap,
                'mean_auc': mean_auc,
                'mean_brier': mean_brier,
                'mean_downstream_cv': mean_downstream_cv,
                'model_score': model_score,
                'std_ap': np.std([s['ap'] for s in fold_scores]),
                'n_valid_folds': len(fold_scores),
            })

    if best_params is None and results:
        fallback = max(results, key=lambda r: r.get('mean_ap', -np.inf))
        param_keys = ['depth', 'learning_rate', 'l2_leaf_reg', 'min_data_in_leaf', 'random_strength', 'bagging_temperature', 'rsm']
        best_params = {k: fallback[k] for k in param_keys if k in fallback}

    return {
        'best_params': best_params,
        'best_score': best_score,
        'model_selection_mode': model_selection_mode,
        'results_df': pd.DataFrame(results),
        'feature_columns': feature_columns,
        'folds': folds,
        'time_elapsed_sec': time.time() - start_time,
    }


def tune_policy_parameters(
        dataset_df: pd.DataFrame,
        model_params: dict,
        target_col: str = 'target_bad_next_5',
        time_budget_min: int = 30,
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
        embargo_hours: float = 0,
        min_valid_folds: int = 2,
        score_mode: str = 'pnl_improvement',
        policy_grid_preset: str = 'default',
        feature_profile: str = None,
) -> dict:
    start_time = time.time()
    time_budget_sec = time_budget_min * 60

    feature_columns = get_regime_feature_columns(dataset_df, feature_profile=feature_profile)

    folds = generate_regime_walk_forward_folds(
        dataset_df, fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
    )

    if not folds:
        raise ValueError("Not enough data to generate walk-forward folds")

    policy_grid = get_policy_parameter_grid(preset=policy_grid_preset)

    fold_models = []
    for fold in folds:
        fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals, embargo_hours=embargo_hours)
        model, val_df = train_regime_fold(
            fold_df, feature_columns, target_col, model_params,
            iterations=iterations, early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        fold_models.append((model, val_df))

    best_score = -np.inf
    best_params = None
    best_fold_results = []
    fallback_score = -np.inf
    fallback_params = None
    fallback_fold_results = []
    results = []

    for policy_params in policy_grid:
        elapsed = time.time() - start_time
        if elapsed >= time_budget_sec:
            break

        fold_results = []
        for model, val_df in fold_models:
            eval_result = evaluate_regime_fold(
                model, val_df, feature_columns, policy_params,
                max_blocked_share=max_blocked_share,
                min_signal_keep_rate=min_signal_keep_rate,
                score_mode=score_mode,
                target_col=target_col,
            )
            if eval_result.get('valid', False):
                fold_results.append(eval_result)

        if len(fold_results) >= min_valid_folds:
            scores = [r['score'] for r in fold_results]
            weights = [r.get('signals_before', 1) for r in fold_results]
            weighted_mean = np.average(scores, weights=weights) if sum(weights) > 0 else np.mean(scores)

            n_no_op = sum(1 for r in fold_results if r.get('no_op', False))
            final_score = weighted_mean

            if final_score > best_score:
                best_score = final_score
                best_params = policy_params
                best_fold_results = fold_results

            results.append({
                **{f'policy_{k}': v for k, v in policy_params.items()},
                'score': final_score,
                'n_valid_folds': len(fold_results),
                'n_no_op_folds': n_no_op,
            })
        elif fold_results:
            scores = [r['score'] for r in fold_results]
            weights = [r.get('signals_before', 1) for r in fold_results]
            weighted_mean = np.average(scores, weights=weights) if sum(weights) > 0 else np.mean(scores)
            if (
                    fallback_params is None or
                    len(fold_results) > len(fallback_fold_results) or
                    (len(fold_results) == len(fallback_fold_results) and weighted_mean > fallback_score)
            ):
                fallback_score = weighted_mean
                fallback_params = policy_params
                fallback_fold_results = fold_results

    if best_params is None:
        if fallback_params is not None:
            best_params = fallback_params
            best_score = fallback_score
            best_fold_results = fallback_fold_results
        elif policy_grid:
            best_params = policy_grid[0]
            best_score = -np.inf
            best_fold_results = []

    resolved_thresholds = None
    if best_params and best_fold_results:
        resolved_pause = [r['resolved_pause_threshold'] for r in best_fold_results if 'resolved_pause_threshold' in r]
        resolved_resume = [r['resolved_resume_threshold'] for r in best_fold_results if 'resolved_resume_threshold' in r]
        if resolved_pause and resolved_resume:
            resolved_thresholds = {
                'pause_on_threshold': float(np.median(resolved_pause)),
                'resume_threshold': float(np.median(resolved_resume)),
                'resume_confirm_signals': best_params['resume_confirm_signals'],
            }

    return {
        'best_params': best_params,
        'best_score': best_score,
        'results_df': pd.DataFrame(results),
        'time_elapsed_sec': time.time() - start_time,
        'fold_results': best_fold_results,
        'resolved_thresholds': resolved_thresholds,
    }


def _compute_cv_summary(fold_results: list) -> dict:
    if not fold_results:
        return {
            'mean_score': -np.inf,
            'std_score': 0.0,
            'n_valid_folds': 0,
            'n_no_op_folds': 0,
            'fold_results': [],
            'mean_pnl_improvement': 0.0,
            'mean_blocked_share': 0.0,
            'median_blocked_share': 0.0,
            'positive_improvement_ratio': 0.0,
        }

    scores = [r.get('score', -np.inf) for r in fold_results]
    weights = [r.get('signals_before', 1) for r in fold_results]

    weighted_mean_score = np.average(scores, weights=weights) if sum(weights) > 0 else np.mean(scores)
    std_score = np.std(scores) if len(scores) > 1 else 0.0

    n_no_op = sum(1 for r in fold_results if r.get('no_op', False))
    pnl_improvements = [r.get('pnl_improvement', 0) for r in fold_results]
    blocked_shares = [r.get('blocked_share', 0) for r in fold_results]

    positive_improvements = sum(1 for p in pnl_improvements if p > 0)
    positive_improvement_ratio = positive_improvements / len(fold_results) if fold_results else 0

    return {
        'mean_score': weighted_mean_score,
        'std_score': std_score,
        'n_valid_folds': len(fold_results),
        'n_no_op_folds': n_no_op,
        'fold_results': fold_results,
        'mean_pnl_improvement': np.mean(pnl_improvements) if pnl_improvements else 0.0,
        'mean_blocked_share': np.mean(blocked_shares) if blocked_shares else 0.0,
        'median_blocked_share': np.median(blocked_shares) if blocked_shares else 0.0,
        'positive_improvement_ratio': positive_improvement_ratio,
    }


def tune_regime_guard(
        dataset_df: pd.DataFrame,
        target_col: str = 'target_bad_next_5',
        time_budget_min: int = 90,
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
        embargo_hours: float = 0,
        min_valid_folds: int = 2,
        score_mode: str = 'pnl_improvement',
        policy_grid_preset: str = 'default',
        model_selection_mode: str = 'downstream_cv',
        feature_profile: str = None,
) -> dict:
    model_budget = int(time_budget_min * 0.5)
    policy_budget = int(time_budget_min * 0.5)

    model_result = tune_model_hyperparameters(
        dataset_df, target_col=target_col,
        time_budget_min=model_budget,
        fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
        iterations=iterations, early_stopping_rounds=early_stopping_rounds,
        seed=seed, embargo_signals=embargo_signals, embargo_hours=embargo_hours,
        min_valid_folds=min_valid_folds,
        max_blocked_share=max_blocked_share,
        min_signal_keep_rate=min_signal_keep_rate,
        score_mode=score_mode,
        model_selection_mode=model_selection_mode,
        feature_profile=feature_profile,
        policy_grid_preset=policy_grid_preset,
    )

    if model_result['best_params'] is None:
        raise ValueError("Failed to find valid model parameters")

    policy_result = tune_policy_parameters(
        dataset_df, model_result['best_params'],
        target_col=target_col,
        time_budget_min=policy_budget,
        fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
        iterations=iterations, early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        max_blocked_share=max_blocked_share,
        min_signal_keep_rate=min_signal_keep_rate,
        embargo_signals=embargo_signals, embargo_hours=embargo_hours,
        min_valid_folds=min_valid_folds,
        score_mode=score_mode,
        policy_grid_preset=policy_grid_preset,
        feature_profile=feature_profile,
    )

    if policy_result['best_params'] is None:
        raise ValueError("Failed to find valid policy parameters")

    model_leaderboard_df = model_result.get('results_df', pd.DataFrame())
    policy_leaderboard_df = policy_result.get('results_df', pd.DataFrame())

    model_trials = len(model_leaderboard_df) if not model_leaderboard_df.empty else 0
    policy_trials = len(policy_leaderboard_df) if not policy_leaderboard_df.empty else 0
    total_trials = model_trials + policy_trials
    total_time = model_result.get('time_elapsed_sec', 0) + policy_result.get('time_elapsed_sec', 0)

    return {
        'best_model_params': model_result['best_params'],
        'best_policy_params': policy_result['best_params'],
        'best_score': policy_result.get('best_score', -np.inf),
        'model_selection_mode': model_result.get('model_selection_mode', model_selection_mode),
        'best_cv_result': _compute_cv_summary(policy_result.get('fold_results', [])),
        'model_leaderboard': model_leaderboard_df,
        'policy_leaderboard': policy_leaderboard_df,
        'feature_columns': model_result['feature_columns'],
        'folds': model_result['folds'],
        'trials_completed': total_trials,
        'time_elapsed_sec': total_time,
        'model_tuning': model_result,
        'policy_tuning': policy_result,
    }


def train_final_regime_model(
        dataset_df: pd.DataFrame,
        feature_columns: list,
        target_col: str,
        params: dict,
        train_end: datetime,
        iterations: int = 10000,
        seed: int = 42,
        target_horizon_signals: int = 5,
        embargo_hours: float = 0,
        auto_class_weights: str = None,
) -> CatBoostClassifier:
    train_df = dataset_df[dataset_df['open_time'] < train_end]

    train_df = train_df.dropna(subset=[target_col])

    train_df = train_df.sort_values('open_time')
    if len(train_df) > target_horizon_signals:
        train_df = train_df.iloc[:-target_horizon_signals]

    if embargo_hours > 0:
        embargo_cutoff = train_end - pd.Timedelta(hours=embargo_hours)
        train_df = train_df[train_df['open_time'] < embargo_cutoff]

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    w_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None

    if w_train is not None:
        valid_mask = ~np.isnan(w_train)
        if not valid_mask.all():
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            w_train = w_train[valid_mask]

    train_pool = Pool(X_train, y_train, weight=w_train)

    catboost_params = {
        'iterations': iterations,
        'depth': params['depth'],
        'learning_rate': params['learning_rate'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'random_strength': params.get('random_strength', 1.0),
        'bagging_temperature': params.get('bagging_temperature', 1.0),
        'rsm': params.get('rsm', 1.0),
        'random_seed': seed,
        'verbose': 100,
        'eval_metric': 'Logloss',
    }

    if auto_class_weights and w_train is None:
        catboost_params['auto_class_weights'] = auto_class_weights

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool)
    return model
