import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pump_end_threshold.ml.regime_evaluate import (
    evaluate_regime,
    compute_regime_scorecard,
    build_pause_episodes,
    build_bucket_summary_6h,
    build_p_bad_deciles,
    save_btc_pause_overlay,
    build_pause_start_explanations,
)
from pump_end_threshold.ml.regime_feature_schema import get_regime_feature_columns
from pump_end_threshold.ml.regime_tuning import (
    tune_regime_guard,
    train_final_regime_model,
    resolve_policy_params,
)
from pump_end_threshold.ml.regime_policy import RegimePolicy
from pump_end_threshold.ml.train import get_feature_importance


def log(level: str, component: str, message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level}] {timestamp} [{component}] {message}")


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt


def _feature_family(name: str) -> str:
    if name.startswith('token_vs_'):
        return 'token_vs'
    if name.startswith(('token_vol_spike_relative',)):
        return 'token_vs'
    if name.startswith('token_'):
        return 'token'
    if name.startswith('btc_eth_'):
        return 'btc_eth'
    if name.startswith('btc_'):
        return 'btc'
    if name.startswith('eth_'):
        return 'eth'
    if name.startswith('breadth_'):
        return 'breadth'
    if name.startswith('raw_signals_'):
        return 'raw_signals'
    if name.startswith('unique_symbols_'):
        return 'unique_symbols'
    if name.startswith('signal_density_'):
        return 'signal_density'
    if name.startswith('max_symbol_'):
        return 'unique_symbols'
    if name.startswith('bucket_'):
        return 'bucket'
    if name.startswith('strat_'):
        return 'strat'
    return 'other'


def main():
    parser = argparse.ArgumentParser(description="Train regime guard model")
    parser.add_argument("--dataset-parquet", type=str, required=True,
                        help="Path to regime dataset parquet")
    parser.add_argument("--target-col", type=str, default="target_pause_value_next_12h")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Train end date (YYYY-MM-DD), exclusive. If set, trains final model")
    parser.add_argument("--time-budget-min", type=int, default=60)
    parser.add_argument("--fold-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--fold-days", type=int, default=14,
                        help="Val fold size in days (overrides --fold-months if set)")
    parser.add_argument("--min-train-days", type=int, default=None,
                        help="Min training period in days (overrides --min-train-months if set)")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embargo-signals", type=int, default=5,
                        help="Number of signals to embargo at train/val boundary")
    parser.add_argument("--embargo-hours", type=float, default=12,
                        help="Hours of time-based embargo at train/val boundary (for time-based targets)")
    parser.add_argument("--max-blocked-share", type=float, default=0.35)
    parser.add_argument("--min-signal-keep-rate", type=float, default=0.45)
    parser.add_argument("--min-valid-folds", type=int, default=2,
                        help="Minimum number of valid folds required")
    parser.add_argument("--score-mode", type=str, default="pnl_improvement",
                        choices=["pnl_after", "pnl_improvement", "block_value", "comprehensive"],
                        help="Scoring mode for hyperparameter optimization")
    parser.add_argument("--policy-grid", type=str, default="default",
                        choices=["default", "conservative", "aggressive", "low"],
                        help="Policy parameter grid preset")
    parser.add_argument("--disable-auto-class-weights", action="store_true",
                        help="Disable auto_class_weights in CatBoost (use manual sample_weight instead)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directory for all artifacts (if not provided, creates timestamped dir)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Parent directory for run-dir (deprecated, use --run-dir)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for subdirectory (deprecated, use --run-dir)")
    parser.add_argument("--clickhouse-dsn", type=str, default=None,
                        help="ClickHouse DSN for BTC price chart generation (optional)")

    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.out_dir:
        if args.run_name:
            run_dir = Path(args.out_dir) / args.run_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = Path(args.out_dir) / f"regime_run_{timestamp}"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(f"regime_run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    log("INFO", "REGIME", f"run_dir={run_dir}")

    log("INFO", "REGIME", f"loading dataset from {args.dataset_parquet}")
    dataset = pd.read_parquet(args.dataset_parquet)
    log("INFO", "REGIME", f"loaded {len(dataset)} rows")

    feature_columns = get_regime_feature_columns(dataset)
    log("INFO", "REGIME", f"feature columns: {len(feature_columns)}")

    if args.target_col not in dataset.columns:
        available_targets = [col for col in dataset.columns if col.startswith('target_')]
        log("ERROR", "REGIME", f"Target column '{args.target_col}' not in dataset")
        log("ERROR", "REGIME", f"Available target columns: {available_targets}")
        log("ERROR", "REGIME", f"Total columns in dataset: {len(dataset.columns)}")
        if len(dataset.columns) < 50:
            log("ERROR", "REGIME", f"Dataset appears to be incomplete. Columns: {list(dataset.columns)}")
        raise ValueError(f"Target column '{args.target_col}' not in dataset. Dataset may need to be rebuilt with updated code.")

    target_rate = dataset[args.target_col].mean()
    log("INFO", "REGIME", f"target '{args.target_col}' rate: {target_rate:.3f}")

    config = vars(args)
    config['n_features'] = len(feature_columns)
    config['n_samples'] = len(dataset)
    config['target_rate'] = float(target_rate)
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)

    train_end = parse_date_exclusive(args.train_end) if args.train_end else None

    if train_end:
        cv_data = dataset[dataset['open_time'] < train_end].copy()
        log("INFO", "REGIME", f"CV data: {len(cv_data)} rows (before {args.train_end})")
    else:
        cv_data = dataset

    log("INFO", "REGIME", f"starting tuning with time_budget={args.time_budget_min}min")
    tune_result = tune_regime_guard(
        cv_data,
        target_col=args.target_col,
        time_budget_min=args.time_budget_min,
        fold_months=args.fold_months,
        min_train_months=args.min_train_months,
        fold_days=args.fold_days,
        min_train_days=args.min_train_days,
        iterations=args.iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        embargo_signals=args.embargo_signals,
        embargo_hours=args.embargo_hours,
        max_blocked_share=args.max_blocked_share,
        min_signal_keep_rate=args.min_signal_keep_rate,
        min_valid_folds=args.min_valid_folds,
        score_mode=args.score_mode,
        policy_grid_preset=args.policy_grid,
    )

    log("INFO", "REGIME",
        f"tuning done: {tune_result['trials_completed']} trials in {tune_result['time_elapsed_sec']:.1f}s")
    log("INFO", "REGIME", f"best score: {tune_result['best_score']:.4f}")
    log("INFO", "REGIME", f"best model params: {tune_result['best_model_params']}")
    log("INFO", "REGIME", f"best policy params: {tune_result['best_policy_params']}")

    with open(run_dir / "best_model_params.json", 'w') as f:
        json.dump(tune_result['best_model_params'], f, indent=2)

    with open(run_dir / "best_policy_params.json", 'w') as f:
        json.dump(tune_result['best_policy_params'], f, indent=2)

    tune_result['model_leaderboard'].to_csv(run_dir / "model_leaderboard.csv", index=False)
    tune_result['policy_leaderboard'].to_csv(run_dir / "policy_leaderboard.csv", index=False)

    with open(run_dir / "folds.json", 'w') as f:
        json.dump(tune_result['folds'], f, indent=2, default=str)

    cv_result = tune_result['best_cv_result']
    if cv_result:
        with open(run_dir / "cv_report.json", 'w') as f:
            serializable = {
                'mean_score': cv_result.get('mean_score'),
                'std_score': cv_result.get('std_score'),
                'n_valid_folds': cv_result.get('n_valid_folds'),
                'fold_results': cv_result.get('fold_results', []),
            }
            json.dump(serializable, f, indent=2, default=str)

        if cv_result.get('fold_results'):
            fold_metrics = pd.DataFrame(cv_result['fold_results'])
            fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)

    with open(run_dir / "policy_report.json", 'w') as f:
        policy_report = {
            'policy_params': tune_result['best_policy_params'],
            'resolved_thresholds': tune_result.get('policy_tuning', {}).get('resolved_thresholds'),
        }
        json.dump(policy_report, f, indent=2, default=str)

    if train_end:
        max_open_time_all = dataset['open_time'].max()
        target_valid = dataset[dataset[args.target_col].notna()]
        max_open_time_target = target_valid['open_time'].max() if len(target_valid) > 0 else None
        test_df_check = dataset[dataset['open_time'] >= train_end]
        n_test_rows = len(test_df_check)
        n_test_with_target = len(test_df_check[test_df_check[args.target_col].notna()])

        log("INFO", "REGIME", f"max_open_time={max_open_time_all}, max_open_time_with_target={max_open_time_target}")
        log("INFO", "REGIME", f"n_test_rows={n_test_rows}, n_test_with_target={n_test_with_target}")

        if n_test_with_target == 0:
            raise ValueError(
                f"No test rows with valid target after train_end={train_end}. "
                f"Dataset max_open_time={max_open_time_all}, "
                f"max_open_time_with_target={max_open_time_target}. "
                f"Run is invalid without real holdout."
            )

        log("INFO", "REGIME", f"training final model on data before {args.train_end}")

        final_model = train_final_regime_model(
            dataset,
            tune_result['feature_columns'],
            args.target_col,
            tune_result['best_model_params'],
            train_end,
            iterations=args.iterations,
            seed=args.seed,
            embargo_hours=args.embargo_hours,
        )

        model_path = run_dir / "regime_guard_model.cbm"
        final_model.save_model(str(model_path))
        log("INFO", "REGIME", f"model saved to {model_path}")

        importance_df = get_feature_importance(final_model, tune_result['feature_columns'])
        importance_df.to_csv(run_dir / "feature_importance.csv", index=False)

        grouped_importance = importance_df.copy()
        grouped_importance['feature_group'] = grouped_importance['feature'].apply(_feature_family)
        grouped_summary = grouped_importance.groupby('feature_group')['importance'].agg(['sum', 'mean', 'count'])
        grouped_summary = grouped_summary.sort_values('sum', ascending=False)
        grouped_summary.to_csv(run_dir / "feature_importance_grouped.csv")

        test_df = dataset[dataset['open_time'] >= train_end].copy()
        if len(test_df) > 0:
            X_test = test_df[tune_result['feature_columns']]
            p_bad_test = final_model.predict_proba(X_test)[:, 1]
            test_df['p_bad'] = p_bad_test

            resolved = resolve_policy_params(tune_result['best_policy_params'], p_bad_test)
            policy = RegimePolicy(**resolved)
            filtered = policy.apply(test_df, p_bad_col='p_bad')

            scorecard = compute_regime_scorecard(filtered)
            test_metrics = evaluate_regime(filtered, target_col=args.target_col)

            log("INFO", "REGIME",
                f"test pnl_improvement={scorecard['pnl_improvement']:.1f} "
                f"blocked_bad_precision={scorecard['blocked_bad_precision']:.2f} "
                f"sl_capture={scorecard['sl_capture']:.2f} "
                f"tp_tax={scorecard['tp_tax']:.2f} "
                f"worst_12h_improvement={scorecard['worst_window_improvement_12h']:.1f}")

            with open(run_dir / "test_scorecard.json", 'w') as f:
                json.dump(scorecard, f, indent=2, default=str)

            with open(run_dir / "test_metrics_full.json", 'w') as f:
                json.dump(test_metrics, f, indent=2, default=str)

            filtered.to_parquet(run_dir / "test_predictions.parquet", index=False)
            filtered.to_parquet(run_dir / "test_scored.parquet", index=False)

            accepted = filtered[~filtered['blocked_by_policy']]
            blocked = filtered[filtered['blocked_by_policy']]
            accepted.to_parquet(run_dir / "test_accepted.parquet", index=False)
            blocked.to_parquet(run_dir / "test_blocked.parquet", index=False)

            if not blocked.empty and 'symbol' in blocked.columns and 'open_time' in blocked.columns:
                blocked_signals_csv = blocked[['symbol', 'open_time']].copy()
                blocked_signals_csv = blocked_signals_csv.rename(columns={'open_time': 'timestamp'})
                blocked_signals_csv = blocked_signals_csv.sort_values(['timestamp', 'symbol'])
                blocked_signals_csv = blocked_signals_csv.drop_duplicates(subset=['symbol', 'timestamp'])
                blocked_signals_csv.to_csv(run_dir / "test_blocked_signals.csv", index=False)
                log("INFO", "REGIME", f"Saved {len(blocked_signals_csv)} blocked signals to test_blocked_signals.csv")

            episodes_df = build_pause_episodes(filtered)
            if not episodes_df.empty:
                episodes_df.to_csv(run_dir / "pause_episodes.csv", index=False)
                log("INFO", "REGIME", f"Saved {len(episodes_df)} pause episodes")

            bucket_df = build_bucket_summary_6h(filtered)
            if not bucket_df.empty:
                bucket_df.to_csv(run_dir / "bucket_summary_6h.csv", index=False)

            deciles_df = build_p_bad_deciles(filtered)
            if not deciles_df.empty:
                deciles_df.to_csv(run_dir / "p_bad_deciles.csv", index=False)

            if args.clickhouse_dsn:
                try:
                    from pump_end_threshold.infra.clickhouse import DataLoader
                    btc_loader = DataLoader(args.clickhouse_dsn)
                    t_min = filtered['open_time'].min() - timedelta(hours=6)
                    t_max = filtered['open_time'].max() + timedelta(hours=6)
                    btc_candles = btc_loader.load_candles_range('BTCUSDT', t_min, t_max)
                    save_btc_pause_overlay(filtered, btc_candles, run_dir / "btc_pause_overlay_6h.png")
                    log("INFO", "REGIME", "Saved btc_pause_overlay_6h.png")
                except Exception as e:
                    log("WARN", "REGIME", f"Could not generate BTC overlay chart: {e}")

            explanations_df = build_pause_start_explanations(
                filtered, final_model, tune_result['feature_columns'],
            )
            if not explanations_df.empty:
                explanations_df.to_csv(run_dir / "pause_start_explanations.csv", index=False)
                log("INFO", "REGIME", f"Saved {len(explanations_df)} pause start explanations")

            if 'open_time' in filtered.columns:
                monthly_results = []
                filtered_monthly = filtered.copy()
                filtered_monthly['month'] = pd.to_datetime(filtered_monthly['open_time']).dt.to_period('M')
                for month, month_df in filtered_monthly.groupby('month'):
                    accepted_month = month_df[~month_df['blocked_by_policy']]
                    blocked_month = month_df[month_df['blocked_by_policy']]
                    has_outcome = 'trade_outcome' in month_df.columns
                    has_pnl = 'pnl_pct' in month_df.columns

                    row = {
                        'month': str(month),
                        'signals': len(month_df),
                        'signals_after': len(accepted_month),
                        'blocked_share': float(month_df['blocked_by_policy'].mean()),
                    }

                    if has_outcome:
                        row['sl_blocked'] = int((blocked_month['trade_outcome'] == 'SL').sum())
                        row['tp_blocked'] = int((blocked_month['trade_outcome'] == 'TP').sum())

                    if has_pnl:
                        row['pnl_before'] = float(month_df['pnl_pct'].fillna(0).sum())
                        row['pnl_after'] = float(accepted_month['pnl_pct'].fillna(0).sum())
                        row['pnl_improvement'] = row['pnl_after'] - row['pnl_before']

                    monthly_results.append(row)

                if monthly_results:
                    monthly_backtest = pd.DataFrame(monthly_results)
                    monthly_backtest.to_csv(run_dir / "monthly_backtest.csv", index=False)

        with open(run_dir / "feature_columns.json", 'w') as f:
            json.dump(tune_result['feature_columns'], f, indent=2)

    log("INFO", "REGIME", f"done. artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
