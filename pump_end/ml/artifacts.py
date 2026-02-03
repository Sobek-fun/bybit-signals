import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


class RunArtifacts:
    def __init__(self, out_dir: str, run_name: str = None):
        if run_name:
            self.run_dir = Path(out_dir) / run_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = Path(out_dir) / f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir = self.run_dir / "clusters"

    def _ensure_clusters_dir(self):
        self.clusters_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict):
        path = self.run_dir / "run_config.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save_labels_filtered(self, df: pd.DataFrame):
        path = self.run_dir / "pump_labels_filtered.csv"
        df.to_csv(path, index=False)

    def save_training_points(self, df: pd.DataFrame):
        path = self.run_dir / "training_points.parquet"
        df.to_parquet(path, index=False)

    def save_features(self, df: pd.DataFrame):
        path = self.run_dir / "features.parquet"
        df.to_parquet(path, index=False)

    def save_splits(self, splits_info: dict):
        path = self.run_dir / "splits.json"
        with open(path, 'w') as f:
            json.dump(splits_info, f, indent=2, default=str)

    def save_model(self, model):
        path = self.run_dir / "catboost_model.cbm"
        model.save_model(str(path))

    def save_threshold_sweep(self, df: pd.DataFrame):
        path = self.run_dir / "threshold_sweep.csv"
        df.to_csv(path, index=False)

    def save_metrics(self, metrics: dict, split_name: str):
        path = self.run_dir / f"metrics_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_feature_importance(self, df: pd.DataFrame):
        path = self.run_dir / "feature_importance.csv"
        df.to_csv(path, index=False)
        path_detail = self.run_dir / "feature_importance_detail.csv"
        df.to_csv(path_detail, index=False)

    def save_feature_importance_grouped(self, df: pd.DataFrame):
        path = self.run_dir / "feature_importance_grouped.csv"
        df.to_csv(path, index=False)
        path_typo = self.run_dir / "feature_importance_gruped.csv"
        df.to_csv(path_typo, index=False)

    def save_predicted_signals(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_holdout.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals_val(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_val.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals_test(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_test.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals_pool(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_val_test_pool.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals_raw(self, df: pd.DataFrame, split: str):
        path = self.run_dir / f"predicted_signals_{split}_raw.csv"
        df.to_csv(path, index=False)

    def save_predictions(self, df: pd.DataFrame, split_name: str):
        path = self.run_dir / f"predictions_{split_name}.parquet"
        df.to_parquet(path, index=False)

    def save_best_params(self, params: dict):
        path = self.run_dir / "best_params.json"
        with open(path, 'w') as f:
            json.dump(params, f, indent=2, default=str)

    def save_best_threshold(self, threshold: float, signal_rule_params: dict = None):
        data = {'threshold': threshold}
        if signal_rule_params:
            data.update(signal_rule_params)
        path = self.run_dir / "best_threshold.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_leaderboard(self, df: pd.DataFrame):
        path = self.run_dir / "leaderboard.csv"
        df.to_csv(path, index=False)

    def save_cv_report(self, cv_result: dict):
        path = self.run_dir / "cv_report.json"
        serializable = {
            'mean_score': cv_result.get('mean_score'),
            'std_score': cv_result.get('std_score'),
            'mean_threshold': cv_result.get('mean_threshold'),
            'fold_results': cv_result.get('fold_results', [])
        }
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

    def save_folds(self, folds: list):
        path = self.run_dir / "folds.json"
        with open(path, 'w') as f:
            json.dump(folds, f, indent=2, default=str)

    def save_trade_quality(self, metrics: dict, split_name: str):
        path = self.run_dir / f"trade_quality_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_backtest_opt_val(self, data: dict):
        path = self.run_dir / "backtest_opt_val.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_backtest_eval_test(self, data: dict):
        path = self.run_dir / "backtest_eval_test.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_backtest_summary(self, data: dict):
        path = self.run_dir / "backtest_summary.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_offset_distribution(self, data: dict, split_name: str):
        path = self.run_dir / f"offset_distribution_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_cluster_config(self, config: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_config.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save_cluster_model(self, clusterer):
        from pump_end.ml.clustering import save_clusterer
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_model.joblib"
        save_clusterer(clusterer, str(path))

    def save_event_clusters(self, df: pd.DataFrame):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "event_clusters.csv"
        df.to_csv(path, index=False)

    def save_cluster_quality(self, quality: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_quality.json"
        with open(path, 'w') as f:
            json.dump(quality, f, indent=2, default=str)

    def save_cluster_feature_summary(self, df: pd.DataFrame):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_feature_summary.csv"
        df.to_csv(path, index=False)

    def save_cluster_examples(self, df: pd.DataFrame):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_examples.csv"
        df.to_csv(path, index=False)

    def save_cluster_drift_by_month(self, df: pd.DataFrame):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_drift_by_month.csv"
        df.to_csv(path, index=False)

    def save_thresholds_by_cluster(self, thresholds: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "threshold_by_cluster.json"
        with open(path, 'w') as f:
            json.dump(thresholds, f, indent=2, default=str)

    def save_threshold_sweep_by_cluster(self, cluster_id: int, df: pd.DataFrame):
        self._ensure_clusters_dir()
        path = self.clusters_dir / f"threshold_sweep_cluster_{cluster_id}.csv"
        df.to_csv(path, index=False)

    def save_metrics_by_cluster(self, metrics: dict, split_name: str):
        self._ensure_clusters_dir()
        path = self.clusters_dir / f"metrics_by_cluster_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_signals_by_cluster(self, cluster_id: int, df: pd.DataFrame, split_name: str):
        self._ensure_clusters_dir()
        path = self.clusters_dir / f"signals_cluster_{cluster_id}_{split_name}.csv"
        df.to_csv(path, index=False)

    def save_early_continuation_stats(self, stats: dict, split_name: str):
        self._ensure_clusters_dir()
        path = self.clusters_dir / f"early_continuation_stats_{split_name}.json"
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

    def save_signals_with_mae_before_end(self, df: pd.DataFrame, split_name: str):
        self._ensure_clusters_dir()
        path = self.clusters_dir / f"signals_with_mae_before_end_{split_name}.csv"
        df.to_csv(path, index=False)

    def save_cluster_features_dropped(self, dropped: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_features_dropped.json"
        with open(path, 'w') as f:
            json.dump(dropped, f, indent=2, default=str)

    def save_cluster_selectivity_report(self, report: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "cluster_selectivity_report.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def save_trade_quality_by_cluster(self, metrics: dict):
        self._ensure_clusters_dir()
        path = self.clusters_dir / "trade_quality_by_cluster.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def get_path(self) -> Path:
        return self.run_dir

    def get_clusters_path(self) -> Path:
        self._ensure_clusters_dir()
        return self.clusters_dir
