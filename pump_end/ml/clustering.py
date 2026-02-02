import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

CLUSTER_FEATURES = [
    'cum_ret_5', 'cum_ret_10', 'cum_ret_30',
    'ret_1_mean_30', 'ret_1_std_30', 'ret_1_slope_5',
    'vol_ratio_mean_30', 'vol_ratio_std_30', 'vol_ratio_slope_5', 'vol_ratio_max_10',
    'drawdown_last_minus_max_30',
    'close_pos_lag_0', 'wick_ratio_lag_0', 'signed_body_lag_0',
    'max_upper_wick_last_5', 'count_red_last_5',
    'climax_vr_lag_0', 'range_over_atr_lag_0', 'upper_wick_over_atr_lag_0',
    'rsi_14_mean_30', 'rsi_14_max_30', 'rsi_14_minus_corridor',
    'mfi_14_mean_30', 'mfi_14_max_30', 'mfi_14_minus_corridor',
    'macdh_12_26_9_mean_30', 'macdh_12_26_9_max_30', 'macdh_12_26_9_minus_corridor',
    'liq_sweep_flag_lag_0', 'liq_sweep_overshoot_lag_0', 'liq_reject_strength_lag_0',
    'liq_level_dist', 'dist_to_eqh', 'dist_to_pdh', 'dist_to_pwh',
    'runup', 'near_peak', 'vol_fade', 'rsi_fade', 'macd_fade',
    'pump_score', 'predump_peak', 'blowoff_exhaustion'
]

NAN_RATE_THRESHOLD = 0.10


class EventClusterer:
    def __init__(self, k: int = 5, n_components: int = 10, random_state: int = 42):
        self.k = k
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.cluster_features = None
        self.fitted = False

    def fit(self, X: np.ndarray, cluster_features: list):
        self.cluster_features = cluster_features
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
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)
        return self.kmeans.predict(X_pca)

    def transform(self, X: np.ndarray) -> np.ndarray:
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


def get_available_cluster_features(features_df: pd.DataFrame) -> list:
    available = []
    for f in CLUSTER_FEATURES:
        if f in features_df.columns:
            available.append(f)
    return available


def filter_cluster_features_by_nan_rate(
        features_df: pd.DataFrame,
        cluster_features: list,
        train_end,
        nan_rate_threshold: float = NAN_RATE_THRESHOLD
) -> tuple:
    offset_zero = features_df[features_df['offset'] == 0].copy()
    train_events = offset_zero[offset_zero['open_time'] < train_end]

    if len(train_events) == 0:
        return cluster_features, {}

    used_features = []
    dropped_features = {}

    for feat in cluster_features:
        if feat not in train_events.columns:
            dropped_features[feat] = {'nan_rate': 1.0, 'reason': 'column_not_found'}
            continue

        nan_rate = train_events[feat].isna().mean()
        if nan_rate > (1 - nan_rate_threshold):
            dropped_features[feat] = {'nan_rate': float(nan_rate), 'reason': 'high_nan_rate'}
        else:
            used_features.append(feat)

    return used_features, dropped_features


def fit_event_clusterer(
        features_df: pd.DataFrame,
        train_end,
        cluster_features: list = None,
        k: int = 5,
        n_components: int = 10,
        random_state: int = 42
) -> tuple:
    if cluster_features is None:
        cluster_features = get_available_cluster_features(features_df)

    cluster_features, dropped_features = filter_cluster_features_by_nan_rate(
        features_df, cluster_features, train_end
    )

    offset_zero = features_df[features_df['offset'] == 0].copy()
    train_events = offset_zero[offset_zero['open_time'] < train_end]

    if len(train_events) < k:
        raise ValueError(f"Not enough training events ({len(train_events)}) for {k} clusters")

    X_train = train_events[cluster_features].values

    clusterer = EventClusterer(k=k, n_components=n_components, random_state=random_state)
    clusterer.fit(X_train, cluster_features)

    labels = clusterer.predict(X_train)
    distances = clusterer.get_distances(X_train)

    event_clusters_df = pd.DataFrame({
        'event_id': train_events['event_id'].values,
        'symbol': train_events['symbol'].values,
        'event_open_time': train_events['open_time'].values,
        'cluster_id': labels,
        'distance_to_centroid': distances[np.arange(len(labels)), labels]
    })

    sorted_distances = np.sort(distances, axis=1)
    event_clusters_df['cluster_confidence'] = sorted_distances[:, 1] - sorted_distances[:, 0]

    X_pca = clusterer.transform(X_train)
    try:
        silhouette = silhouette_score(X_pca, labels)
    except:
        silhouette = None
    try:
        davies_bouldin = davies_bouldin_score(X_pca, labels)
    except:
        davies_bouldin = None

    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    cluster_reports = {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'cluster_sizes': cluster_sizes,
        'n_train_events': len(train_events),
        'n_features': len(cluster_features),
        'n_components': clusterer.pca.n_components_,
        'explained_variance_ratio': clusterer.pca.explained_variance_ratio_.tolist(),
        'cluster_features_used': cluster_features,
        'cluster_features_dropped': dropped_features
    }

    return clusterer, event_clusters_df, cluster_reports


def assign_event_clusters(
        clusterer: EventClusterer,
        features_df: pd.DataFrame,
        cluster_features: list = None
) -> pd.DataFrame:
    if cluster_features is None:
        cluster_features = clusterer.cluster_features

    offset_zero = features_df[features_df['offset'] == 0].copy()

    X = offset_zero[cluster_features].values
    labels = clusterer.predict(X)
    distances = clusterer.get_distances(X)

    event_to_cluster = dict(zip(offset_zero['event_id'].values, labels))
    event_to_distance = dict(zip(offset_zero['event_id'].values, distances[np.arange(len(labels)), labels]))

    sorted_distances = np.sort(distances, axis=1)
    confidence = sorted_distances[:, 1] - sorted_distances[:, 0]
    event_to_confidence = dict(zip(offset_zero['event_id'].values, confidence))

    features_df = features_df.copy()
    features_df['cluster_id'] = features_df['event_id'].map(event_to_cluster)
    features_df['cluster_dist'] = features_df['event_id'].map(event_to_distance)
    features_df['cluster_confidence'] = features_df['event_id'].map(event_to_confidence)

    return features_df


def compute_cluster_feature_summary(
        features_df: pd.DataFrame,
        cluster_features: list
) -> pd.DataFrame:
    offset_zero = features_df[features_df['offset'] == 0].copy()

    if 'cluster_id' not in offset_zero.columns:
        return pd.DataFrame()

    rows = []
    for cluster_id in sorted(offset_zero['cluster_id'].unique()):
        cluster_data = offset_zero[offset_zero['cluster_id'] == cluster_id]
        for feat in cluster_features:
            if feat not in cluster_data.columns:
                continue
            values = cluster_data[feat].dropna()
            if len(values) == 0:
                continue
            rows.append({
                'cluster_id': cluster_id,
                'feature': feat,
                'mean': values.mean(),
                'std': values.std(),
                'p05': values.quantile(0.05),
                'p50': values.quantile(0.50),
                'p95': values.quantile(0.95)
            })

    return pd.DataFrame(rows)


def compute_cluster_examples(
        features_df: pd.DataFrame,
        n_examples: int = 5
) -> pd.DataFrame:
    offset_zero = features_df[features_df['offset'] == 0].copy()

    if 'cluster_id' not in offset_zero.columns or 'cluster_dist' not in offset_zero.columns:
        return pd.DataFrame()

    rows = []
    for cluster_id in sorted(offset_zero['cluster_id'].unique()):
        cluster_data = offset_zero[offset_zero['cluster_id'] == cluster_id]
        closest = cluster_data.nsmallest(n_examples, 'cluster_dist')
        for _, row in closest.iterrows():
            rows.append({
                'cluster_id': cluster_id,
                'event_id': row['event_id'],
                'symbol': row['symbol'],
                'open_time': row['open_time'],
                'distance_to_centroid': row['cluster_dist']
            })

    return pd.DataFrame(rows)


def compute_cluster_drift_by_month(features_df: pd.DataFrame) -> pd.DataFrame:
    offset_zero = features_df[features_df['offset'] == 0].copy()

    if 'cluster_id' not in offset_zero.columns:
        return pd.DataFrame()

    offset_zero['month'] = pd.to_datetime(offset_zero['open_time']).dt.to_period('M')

    pivot = offset_zero.groupby(['month', 'cluster_id']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    rows = []
    for month in pivot_pct.index:
        for cluster_id in pivot_pct.columns:
            rows.append({
                'month': str(month),
                'cluster_id': cluster_id,
                'count': pivot.loc[month, cluster_id],
                'pct': pivot_pct.loc[month, cluster_id]
            })

    return pd.DataFrame(rows)


def save_clusterer(clusterer: EventClusterer, path: str):
    joblib.dump(clusterer, path)


def load_clusterer(path: str) -> EventClusterer:
    return joblib.load(path)
