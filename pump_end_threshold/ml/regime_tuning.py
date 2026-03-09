import random
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


def get_policy_parameter_grid(preset: str = 'default') -> list:
    if preset == 'conservative':
        policy_grid = {
            'pause_on_threshold': [0.65, 0.70, 0.75],
            'resume_threshold': [0.20, 0.25, 0.30],
            'resume_confirm_signals': [2, 3, 4],
        }
    elif preset == 'aggressive':
        policy_grid = {
            'pause_on_threshold': [0.55, 0.60, 0.65],
            'resume_threshold': [0.35, 0.40, 0.45],
            'resume_confirm_signals': [1, 2],
        }
    else:
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
        iterations: int = 10000,
        early_stopping_rounds: int = 300,
        seed: int = 42,
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

    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val, weight=w_val)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        min_data_in_leaf=params['min_data_in_leaf'],
        random_strength=params.get('random_strength', 1.0),
        bagging_temperature=params.get('bagging_temperature', 1.0),
        rsm=params.get('rsm', 1.0),
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
        score_mode: str = 'pnl_improvement',
) -> dict:
    if model is None or val_df is None or len(val_df) == 0:
        return {'score': -np.inf, 'valid': False, 'no_op': True}

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

    # Apply policy to get actual blocking decisions
    val_with_policy = policy.apply(val_scored, p_bad_col='p_bad')
    val_scored['blocked_by_policy'] = val_with_policy['blocked_by_policy']

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

    longest_streak_before = result.get('longest_losing_streak_before', 0)
    longest_streak_after = result.get('longest_losing_streak_after', 0)
    streak_reduction = longest_streak_before - longest_streak_after

    blocked_bad_precision = sl_blocked / max(1, sl_blocked + tp_blocked)

    no_op_penalty = -1000 if blocked_share == 0 else 0

    worst_window_6h_before = 0.0
    worst_window_6h_after = 0.0
    worst_window_12h_before = 0.0
    worst_window_12h_after = 0.0
    episode_recall = 0.0
    n_episodes_detected = 0

    if 'open_time' in val_scored.columns:
        val_sorted = val_scored.sort_values('open_time').reset_index(drop=True)

        for window_hours, key_prefix in [(6, 'worst_window_6h'), (12, 'worst_window_12h')]:
            window = pd.Timedelta(hours=window_hours)

            for start_idx in range(len(val_sorted)):
                start_time = val_sorted.iloc[start_idx]['open_time']
                end_time = start_time + window
                window_trades = val_sorted[(val_sorted['open_time'] >= start_time) &
                                         (val_sorted['open_time'] < end_time)]

                if len(window_trades) > 0:
                    window_pnl_before = window_trades['pnl_pct'].sum()
                    window_after_pnl = window_trades[~window_trades['blocked_by_policy']]['pnl_pct'].sum()

                    if key_prefix == 'worst_window_6h':
                        worst_window_6h_before = min(worst_window_6h_before, window_pnl_before)
                        worst_window_6h_after = min(worst_window_6h_after, window_after_pnl)
                    else:
                        worst_window_12h_before = min(worst_window_12h_before, window_pnl_before)
                        worst_window_12h_after = min(worst_window_12h_after, window_after_pnl)

        sl_streaks = []
        current_streak = 0
        for _, trade in val_sorted.iterrows():
            if trade['trade_outcome'] == 'SL':
                current_streak += 1
            else:
                if current_streak >= 3:
                    sl_streaks.append({'start': _, 'length': current_streak})
                current_streak = 0

        if current_streak >= 3:
            sl_streaks.append({'start': len(val_sorted) - 1, 'length': current_streak})

        for streak in sl_streaks:
            start_idx = max(0, streak['start'] - streak['length'])
            if val_sorted.iloc[start_idx:start_idx+2]['blocked_by_policy'].any():
                n_episodes_detected += 1

        if len(sl_streaks) > 0:
            episode_recall = n_episodes_detected / len(sl_streaks)

    worst_window_improvement_6h = worst_window_6h_after - worst_window_6h_before
    worst_window_improvement_12h = worst_window_12h_after - worst_window_12h_before

    if score_mode == 'pnl_after':
        score = pnl_after
    elif score_mode == 'block_value':
        score = block_value
    elif score_mode == 'comprehensive':
        score = (
            pnl_improvement * 1.0 +
            streak_reduction * 20.0 +
            blocked_bad_precision * 50.0 +
            worst_window_improvement_6h * 2.0 +
            worst_window_improvement_12h * 1.0 +
            episode_recall * 100.0 +
            no_op_penalty
        )
        if blocked_bad_precision < 0.35:
            score -= 100
        if tp_block_share_total > 0.30:
            score -= (tp_block_share_total - 0.30) * 100
        if blocked_share > 0.25:
            score -= (blocked_share - 0.25) * 100
    else:
        score = pnl_improvement + worst_window_improvement_12h * 0.5 + streak_reduction * 10.0 + episode_recall * 50.0
        if tp_block_share_total > 0.30:
            score -= (tp_block_share_total - 0.30) * 50
        if blocked_share > 0.25:
            score -= (blocked_share - 0.25) * 100
        score += no_op_penalty

    p_bad_percentiles = np.percentile(p_bad, [50, 90, 95, 99])
    pause_threshold = policy_params['pause_on_threshold']
    signals_above_pause_threshold = np.sum(p_bad >= pause_threshold)
    signals_above_pause_threshold_share = signals_above_pause_threshold / len(p_bad) if len(p_bad) > 0 else 0

    first_hit_timing_signals = []
    if 'signal_id' in val_scored.columns and 'outcome' in val_scored.columns:
        sl_episodes = []
        current_episode = []
        for idx, row in val_scored.iterrows():
            if row['outcome'] == 'sl':
                current_episode.append(idx)
            else:
                if current_episode and len(current_episode) >= 2:
                    sl_episodes.append(current_episode)
                current_episode = []
        if current_episode and len(current_episode) >= 2:
            sl_episodes.append(current_episode)

        for episode in sl_episodes:
            for pos, idx in enumerate(episode):
                if val_scored.loc[idx, 'p_bad'] >= pause_threshold:
                    first_hit_timing_signals.append(pos)
                    break

    avg_first_hit_timing = np.mean(first_hit_timing_signals) if first_hit_timing_signals else -1

    return {
        'score': score,
        'valid': True,
        'no_op': blocked_share == 0,
        'pnl_before': pnl_before,
        'pnl_after': pnl_after,
        'pnl_improvement': pnl_improvement,
        'block_value': block_value,
        'blocked_bad_precision': blocked_bad_precision,
        'longest_streak_reduction': streak_reduction,
        'worst_window_6h_before': worst_window_6h_before,
        'worst_window_6h_after': worst_window_6h_after,
        'worst_window_12h_before': worst_window_12h_before,
        'worst_window_12h_after': worst_window_12h_after,
        'worst_window_improvement_6h': worst_window_improvement_6h,
        'worst_window_improvement_12h': worst_window_improvement_12h,
        'episode_recall': episode_recall,
        'n_episodes_detected': n_episodes_detected,
        'p_bad_p50': p_bad_percentiles[0],
        'p_bad_p90': p_bad_percentiles[1],
        'p_bad_p95': p_bad_percentiles[2],
        'p_bad_p99': p_bad_percentiles[3],
        'signals_above_pause_threshold': signals_above_pause_threshold,
        'signals_above_pause_threshold_share': signals_above_pause_threshold_share,
        'avg_first_hit_timing': avg_first_hit_timing,
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
        score_mode: str = 'pnl_improvement',
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
            score_mode=score_mode,
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
        min_valid_folds: int = 2,
        n_seeds: int = 3,
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
    random.shuffle(model_grid)

    best_score = -np.inf
    best_params = None
    results = []

    default_policy_params = {
        'pause_on_threshold': 0.65,
        'resume_threshold': 0.30,
        'resume_confirm_signals': 2
    }

    for params in model_grid:
        elapsed = time.time() - start_time
        if elapsed >= time_budget_sec:
            break

        fold_scores = []
        regime_scores = []
        for seed_offset in range(n_seeds):
            curr_seed = seed + seed_offset * 123
            for fold_idx, fold in enumerate(folds):
                fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals)
                model, val_df = train_regime_fold(
                    fold_df, feature_columns, target_col, params,
                    iterations=iterations, early_stopping_rounds=early_stopping_rounds,
                    seed=curr_seed,
                )

                if model is None or val_df is None or len(val_df) == 0:
                    continue

                X_val = val_df[feature_columns]
                p_bad = model.predict_proba(X_val)[:, 1]

                from sklearn.metrics import roc_auc_score, log_loss
                y_val = val_df[target_col]
                auc = roc_auc_score(y_val, p_bad)
                logloss = log_loss(y_val, p_bad)
                fold_scores.append({'auc': auc, 'logloss': logloss, 'seed': curr_seed, 'fold': fold_idx})

                regime_eval = evaluate_regime_fold(
                    model, val_df, feature_columns, default_policy_params,
                    max_blocked_share=0.35, min_signal_keep_rate=0.45,
                    score_mode='regime_aware',
                )
                if regime_eval.get('valid', False):
                    regime_scores.append(regime_eval['score'])

        if len(regime_scores) >= min_valid_folds:
            mean_regime_score = np.mean(regime_scores)
            mean_auc = np.mean([s['auc'] for s in fold_scores])
            if mean_regime_score > best_score:
                best_score = mean_regime_score
                best_params = params

            results.append({
                **params,
                'mean_auc': mean_auc,
                'mean_regime_score': mean_regime_score,
                'std_auc': np.std([s['auc'] for s in fold_scores]),
                'n_valid_folds': len(fold_scores),
            })

    return {
        'best_params': best_params,
        'best_score': best_score,
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
        min_valid_folds: int = 2,
        score_mode: str = 'pnl_improvement',
        policy_grid_preset: str = 'default',
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

    policy_grid = get_policy_parameter_grid(preset=policy_grid_preset)

    fold_models = []
    for fold in folds:
        fold_df = apply_regime_fold_split(dataset_df, fold, embargo_signals=embargo_signals)
        model, val_df = train_regime_fold(
            fold_df, feature_columns, target_col, model_params,
            iterations=iterations, early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        fold_models.append((model, val_df))

    best_score = -np.inf
    best_params = None
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

    return {
        'best_params': best_params,
        'best_score': best_score,
        'results_df': pd.DataFrame(results),
        'time_elapsed_sec': time.time() - start_time,
        'fold_results': best_fold_results if 'best_fold_results' in locals() else [],
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
        min_valid_folds: int = 2,
        score_mode: str = 'pnl_improvement',
        policy_grid_preset: str = 'default',
) -> dict:
    model_budget = int(time_budget_min * 0.5)
    policy_budget = int(time_budget_min * 0.5)

    model_result = tune_model_hyperparameters(
        dataset_df, target_col=target_col,
        time_budget_min=model_budget,
        fold_months=fold_months, min_train_months=min_train_months,
        fold_days=fold_days, min_train_days=min_train_days,
        iterations=iterations, early_stopping_rounds=early_stopping_rounds,
        seed=seed, embargo_signals=embargo_signals,
        min_valid_folds=min_valid_folds,
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
        embargo_signals=embargo_signals,
        min_valid_folds=min_valid_folds,
        score_mode=score_mode,
        policy_grid_preset=policy_grid_preset,
    )

    if policy_result['best_params'] is None:
        raise ValueError("Failed to find valid policy parameters")

    # Combine results for backward compatibility
    combined_leaderboard = []

    # Add model results to leaderboard
    if 'results_df' in model_result and not model_result['results_df'].empty:
        for _, row in model_result['results_df'].iterrows():
            combined_leaderboard.append({
                **row.to_dict(),
                'policy_pause_on_threshold': np.nan,
                'policy_resume_threshold': np.nan,
                'policy_resume_confirm_signals': np.nan,
                'mean_score': row.get('mean_auc', np.nan),
                'std_score': row.get('std_auc', np.nan),
            })

    # Add policy results to leaderboard
    if 'results_df' in policy_result and not policy_result['results_df'].empty:
        for _, row in policy_result['results_df'].iterrows():
            record = {
                **model_result['best_params'],
                **row.to_dict(),
                'mean_score': row.get('score', np.nan),
                'std_score': np.nan,
                'n_valid_folds': row.get('n_valid_folds', 0),
            }
            combined_leaderboard.append(record)

    leaderboard_df = pd.DataFrame(combined_leaderboard) if combined_leaderboard else pd.DataFrame()

    # Calculate total trials
    model_trials = len(model_result.get('results_df', [])) if 'results_df' in model_result else 0
    policy_trials = len(policy_result.get('results_df', [])) if 'results_df' in policy_result else 0
    total_trials = model_trials + policy_trials

    # Calculate total time
    total_time = model_result.get('time_elapsed_sec', 0) + policy_result.get('time_elapsed_sec', 0)

    return {
        'best_model_params': model_result['best_params'],
        'best_policy_params': policy_result['best_params'],
        'best_score': policy_result.get('best_score', -np.inf),
        'best_cv_result': _compute_cv_summary(policy_result.get('fold_results', [])),
        'leaderboard': leaderboard_df,
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
) -> CatBoostClassifier:
    train_df = dataset_df[dataset_df['open_time'] < train_end]

    train_df = train_df.dropna(subset=[target_col])

    train_df = train_df.sort_values('open_time')
    if len(train_df) > target_horizon_signals:
        train_df = train_df.iloc[:-target_horizon_signals]

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
        random_strength=params.get('random_strength', 1.0),
        bagging_temperature=params.get('bagging_temperature', 1.0),
        rsm=params.get('rsm', 1.0),
        random_seed=seed,
        verbose=100,
        eval_metric='Logloss',
        auto_class_weights='Balanced',
    )

    model.fit(train_pool)
    return model
