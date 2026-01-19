import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier


class PumpEndModel:
    def __init__(self, model_dir: str):
        model_path = Path(model_dir)

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path / "catboost_model.cbm"))

        self.feature_names = self.model.feature_names_

        with open(model_path / "best_threshold.json", 'r') as f:
            threshold_config = json.load(f)

        self.threshold = threshold_config['threshold']
        self.signal_rule = threshold_config.get('signal_rule', 'pending_turn_down')
        self.min_pending_bars = threshold_config.get('min_pending_bars', 1)
        self.drop_delta = threshold_config.get('drop_delta', 0.0)

        self.window_bars = 30
        self.warmup_bars = 150
        self.feature_set = "base"

        run_config_path = model_path / "run_config.json"
        if run_config_path.exists():
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            self.window_bars = run_config.get('window_bars', 30)
            self.warmup_bars = run_config.get('warmup_bars', 150)
            self.feature_set = run_config.get('feature_set', 'base')

    def predict(self, features_row: dict) -> float:
        feature_values = []
        for name in self.feature_names:
            val = features_row.get(name)
            feature_values.append(np.nan if val is None else val)

        proba = self.model.predict_proba([feature_values])[0][1]
        return proba
