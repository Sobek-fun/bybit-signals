import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

LOG1P_FEATURES = [
    'vol_ratio_mean_30', 'vol_ratio_std_30', 'vol_ratio_slope_5', 'vol_ratio_max_10',
    'climax_vr_lag_0', 'range_over_atr_lag_0', 'upper_wick_over_atr_lag_0',
    'liq_sweep_overshoot_lag_0'
]


class EventClusterer:
    def __init__(self, k: int = 5, n_components: int = 5, random_state: int = 42):
        self.k = k
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.cluster_features = None
        self.fitted = False
        self.log1p_indices = []
        self.clip_lower = None
        self.clip_upper = None

    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        if not self.log1p_indices:
            return X
        X = X.copy()
        for idx in self.log1p_indices:
            X[:, idx] = np.log1p(np.abs(X[:, idx])) * np.sign(X[:, idx])
        return X

    def _apply_clip(self, X: np.ndarray) -> np.ndarray:
        if self.clip_lower is None or self.clip_upper is None:
            return X
        X = X.copy()
        X = np.clip(X, self.clip_lower, self.clip_upper)
        return X

    def _compute_log1p_indices(self, cluster_features: list):
        self.log1p_indices = [
            i for i, f in enumerate(cluster_features) if f in LOG1P_FEATURES
        ]

    def _compute_clip_bounds(self, X: np.ndarray):
        self.clip_lower = np.nanquantile(X, 0.01, axis=0)
        self.clip_upper = np.nanquantile(X, 0.99, axis=0)

    def fit(self, X: np.ndarray, cluster_features: list):
        self.cluster_features = cluster_features
        self._compute_log1p_indices(cluster_features)
        X = self._apply_log1p(X)
        self._compute_clip_bounds(X)
        X = self._apply_clip(X)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        actual_components = min(self.n_components, X_scaled.shape[1], X_scaled.shape[0])
        if actual_components != self.n_components:
            self.pca = PCA(n_components=actual_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        self.kmeans.fit(X_pca)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._apply_log1p(X)
        X = self._apply_clip(X)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)
        return self.kmeans.predict(X_pca)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._apply_log1p(X)
        X = self._apply_clip(X)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def get_distances(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.transform(X)
        distances = np.zeros((X_pca.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X_pca - self.kmeans.cluster_centers_[i], axis=1)
        return distances


def load_clusterer(path: str) -> EventClusterer:
    return joblib.load(path)
