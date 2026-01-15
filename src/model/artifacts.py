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

    def save_feature_importance_grouped(self, df: pd.DataFrame):
        path = self.run_dir / "feature_importance_grouped.csv"
        df.to_csv(path, index=False)

    def save_predicted_signals(self, df: pd.DataFrame):
        path = self.run_dir / "predicted_signals_holdout.csv"
        df.to_csv(path, index=False)

    def save_predictions(self, df: pd.DataFrame, split_name: str):
        path = self.run_dir / f"predictions_{split_name}.parquet"
        df.to_parquet(path, index=False)

    def get_path(self) -> Path:
        return self.run_dir
