import json
import threading
from pathlib import Path

import numpy as np
import joblib
from catboost import CatBoostClassifier

from pump_end_clustering_prod.infra.logging import log
from pump_end.ml.clustering import EventClusterer as _EventClusterer  # noqa: F401


class PumpEndClusteringModel:
    def __init__(self, model_dir: str):
        model_path = Path(model_dir)

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path / "catboost_model.cbm"))

        self.feature_names = self.model.feature_names_

        self._predict_lock = threading.Lock()

        self.window_bars = 30
        self.warmup_bars = 150
        self.feature_set = "base"
        self.neg_before = 20
        self.neg_after = 0
        self.pos_offsets = [0]

        run_config_path = model_path / "run_config.json"
        if run_config_path.exists():
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            self.window_bars = run_config.get('window_bars', 30)
            self.warmup_bars = run_config.get('warmup_bars', 150)
            self.feature_set = run_config.get('feature_set', 'base')
            self.neg_before = run_config.get('neg_before', 20)
            self.neg_after = run_config.get('neg_after', 0)
            pos_offsets_str = run_config.get('pos_offsets', '0')
            self.pos_offsets = [int(x.strip()) for x in str(pos_offsets_str).split(',')]

        with open(model_path / "best_threshold.json", 'r') as f:
            threshold_config = json.load(f)

        self.global_threshold = threshold_config['threshold']
        self.global_signal_rule = threshold_config.get('signal_rule', 'pending_turn_down')
        self.global_min_pending_bars = threshold_config.get('min_pending_bars', 1)
        self.global_drop_delta = threshold_config.get('drop_delta', 0.0)
        self.global_min_pending_peak = threshold_config.get('min_pending_peak', 0.0)
        self.global_min_turn_down_bars = threshold_config.get('min_turn_down_bars', 1)

        self.cluster_mode = False
        self.clusterer = None
        self.cluster_features = []
        self.enabled_clusters = []
        self.thresholds_by_cluster = {}

        cluster_model_path = model_path / "clusters" / "cluster_model.joblib"
        cluster_config_path = model_path / "clusters" / "cluster_config.json"
        cluster_threshold_path = model_path / "clusters" / "threshold_by_cluster.json"

        if cluster_model_path.exists() and cluster_config_path.exists() and cluster_threshold_path.exists():
            try:
                self.clusterer = joblib.load(str(cluster_model_path))

                with open(cluster_config_path, 'r') as f:
                    cluster_config = json.load(f)
                self.cluster_features = cluster_config.get('cluster_features', [])

                with open(cluster_threshold_path, 'r') as f:
                    cluster_threshold = json.load(f)

                self.enabled_clusters = cluster_threshold.get('enabled_clusters', [])

                for key, value in cluster_threshold.items():
                    if key == 'enabled_clusters':
                        continue
                    try:
                        cluster_id = int(key)
                        self.thresholds_by_cluster[cluster_id] = value
                    except (ValueError, TypeError):
                        pass

                if self.enabled_clusters and self.cluster_features:
                    self.cluster_mode = True

                log("INFO", "MODEL",
                    f"cluster mode enabled: enabled_clusters={self.enabled_clusters} "
                    f"cluster_features_count={len(self.cluster_features)} "
                    f"thresholds_by_cluster_keys={list(self.thresholds_by_cluster.keys())}")
            except Exception as e:
                log("ERROR", "MODEL", f"cluster loading failed: {type(e).__name__}: {str(e)}, using global fallback")
                self.cluster_mode = False
        else:
            log("INFO", "MODEL", "cluster artifacts not found, using global fallback")

        log("INFO", "MODEL",
            f"loaded: feature_set={self.feature_set} window_bars={self.window_bars} "
            f"warmup_bars={self.warmup_bars} features_count={len(self.feature_names)}")

        log("INFO", "MODEL",
            f"global params: threshold={self.global_threshold:.4f} "
            f"signal_rule={self.global_signal_rule} "
            f"min_pending_bars={self.global_min_pending_bars} "
            f"drop_delta={self.global_drop_delta} "
            f"min_pending_peak={self.global_min_pending_peak} "
            f"min_turn_down_bars={self.global_min_turn_down_bars}")

        log("INFO", "MODEL", f"cluster_mode={self.cluster_mode}")

    def predict(self, features_row: dict) -> float:
        feature_values = []
        for name in self.feature_names:
            val = features_row.get(name)
            feature_values.append(np.nan if val is None else val)

        with self._predict_lock:
            proba = self.model.predict_proba([feature_values], thread_count=1)[0][1]
        return proba

    def batch_predict(self, feature_rows: list[dict]) -> np.ndarray:
        feature_values_list = []
        for row in feature_rows:
            fv = []
            for name in self.feature_names:
                val = row.get(name)
                fv.append(np.nan if val is None else val)
            feature_values_list.append(fv)
        with self._predict_lock:
            return self.model.predict_proba(feature_values_list, thread_count=1)[:, 1]

    def resolve_params(self, features_row: dict) -> tuple:
        if not self.cluster_mode:
            return None, self._global_params(), True

        cluster_values = []
        for f in self.cluster_features:
            if f not in features_row:
                log("WARN", "MODEL", f"cluster feature missing: {f}, falling back to global")
                return None, self._global_params(), True
            val = features_row[f]
            cluster_values.append(np.nan if val is None else val)

        X = np.array([cluster_values], dtype=np.float64)
        cluster_id = int(self.clusterer.predict(X)[0])

        if cluster_id not in self.enabled_clusters:
            return cluster_id, None, False

        cluster_params = self.thresholds_by_cluster.get(cluster_id)
        if cluster_params is None:
            return cluster_id, None, False

        params = {
            'threshold': cluster_params.get('threshold', self.global_threshold),
            'signal_rule': cluster_params.get('signal_rule', self.global_signal_rule),
            'min_pending_bars': cluster_params.get('min_pending_bars', self.global_min_pending_bars),
            'drop_delta': cluster_params.get('drop_delta', self.global_drop_delta),
            'min_pending_peak': cluster_params.get('min_pending_peak', self.global_min_pending_peak),
            'min_turn_down_bars': cluster_params.get('min_turn_down_bars', self.global_min_turn_down_bars),
        }

        return cluster_id, params, True

    def _global_params(self) -> dict:
        return {
            'threshold': self.global_threshold,
            'signal_rule': self.global_signal_rule,
            'min_pending_bars': self.global_min_pending_bars,
            'drop_delta': self.global_drop_delta,
            'min_pending_peak': self.global_min_pending_peak,
            'min_turn_down_bars': self.global_min_turn_down_bars,
        }
