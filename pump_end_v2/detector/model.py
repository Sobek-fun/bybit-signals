import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pump_end_v2.config import DetectorModelConfig
from pump_end_v2.detector.sequence_dataset import (
    DetectorSequenceStore,
    extract_sequences_for_rows,
)
from pump_end_v2.detector.sequence_model import (
    SequenceTCNBinaryClassifier,
    SequenceTrainStats,
    predict_sequence_model_proba,
    train_sequence_model,
)
from pump_end_v2.features.manifest import (
    DETECTOR_IDENTITY_COLUMNS,
    DETECTOR_SEQUENCE_FEATURE_COLUMNS,
)


@dataclass(slots=True)
class SequenceDetector:
    network: SequenceTCNBinaryClassifier
    feature_columns: tuple[str, ...]
    lookback_bars: int
    random_seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    fit_eval_fraction: float
    fit_eval_min_rows: int
    sequence_learning_rate: float
    weight_decay: float
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    sequence_store: DetectorSequenceStore | None = None
    train_stats: dict[str, float | int | bool | str] = field(default_factory=dict)
    training_history: list[dict[str, float | int]] = field(default_factory=list)

    def save_model(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "network_state_dict": self.network.state_dict(),
            "feature_columns": list(self.feature_columns),
            "lookback_bars": int(self.lookback_bars),
            "random_seed": int(self.random_seed),
            "fit_eval_fraction": float(self.fit_eval_fraction),
            "fit_eval_min_rows": int(self.fit_eval_min_rows),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "train_stats": json.dumps(self.train_stats, ensure_ascii=True),
            "training_history": json.dumps(self.training_history, ensure_ascii=True),
        }
        import torch

        torch.save(payload, str(target))


def build_detector_model(model_config: DetectorModelConfig) -> SequenceDetector:
    import torch

    feature_columns = tuple(DETECTOR_SEQUENCE_FEATURE_COLUMNS)
    input_feature_count = len(feature_columns) + 2
    torch.manual_seed(int(model_config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model_config.random_seed))
    network = SequenceTCNBinaryClassifier(
        input_feature_count=input_feature_count,
        hidden_channels=int(model_config.hidden_channels),
        kernel_size=int(model_config.kernel_size),
        dilations=tuple(int(v) for v in model_config.dilations),
        dropout=float(model_config.dropout),
    )
    lookback_bars = int(model_config.pre_episode_context_bars) + int(
        model_config.decision_window_bars
    )
    return SequenceDetector(
        network=network,
        feature_columns=feature_columns,
        lookback_bars=lookback_bars,
        random_seed=int(model_config.random_seed),
        batch_size=int(model_config.batch_size),
        max_epochs=int(model_config.max_epochs),
        early_stopping_patience=int(model_config.early_stopping_patience),
        fit_eval_fraction=float(model_config.fit_eval_fraction),
        fit_eval_min_rows=int(model_config.fit_eval_min_rows),
        sequence_learning_rate=float(model_config.sequence_learning_rate),
        weight_decay=float(model_config.weight_decay),
    )


def fit_detector_model(
    model: SequenceDetector,
    train_df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    eval_df: pd.DataFrame | None = None,
    sequence_store: DetectorSequenceStore | None = None,
) -> SequenceDetector:
    _require_columns(train_df, ["decision_row_id", target_column], "train_df")
    store = _resolve_sequence_store(model, sequence_store)
    model.sequence_store = store
    x_train_raw, valid_train, in_episode_train = extract_sequences_for_rows(
        store, train_df["decision_row_id"].astype(str)
    )
    y_train = pd.to_numeric(train_df[target_column], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32
    )
    if "target_row_weight" in train_df.columns:
        sample_weight_train = (
            pd.to_numeric(train_df["target_row_weight"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=np.float32)
        )
    else:
        sample_weight_train = np.ones(len(train_df), dtype=np.float32)
    if len(y_train) == 0:
        raise ValueError("fit_detector_model received empty train rows")
    scaler_mean, scaler_std = _fit_scaler(x_train_raw, valid_train)
    x_train_scaled = _transform_with_scaler(x_train_raw, valid_train, scaler_mean, scaler_std)
    x_train_input = _stack_model_inputs(x_train_scaled, valid_train, in_episode_train)
    x_eval_input: np.ndarray | None = None
    y_eval: np.ndarray | None = None
    sample_weight_eval: np.ndarray | None = None
    if eval_df is not None and len(eval_df) > 0:
        _require_columns(eval_df, ["decision_row_id", target_column], "eval_df")
        x_eval_raw, valid_eval, in_episode_eval = extract_sequences_for_rows(
            store, eval_df["decision_row_id"].astype(str)
        )
        y_eval = pd.to_numeric(eval_df[target_column], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
        if "target_row_weight" in eval_df.columns:
            sample_weight_eval = (
                pd.to_numeric(eval_df["target_row_weight"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            sample_weight_eval = np.ones(len(eval_df), dtype=np.float32)
        x_eval_scaled = _transform_with_scaler(
            x_eval_raw, valid_eval, scaler_mean, scaler_std
        )
        x_eval_input = _stack_model_inputs(x_eval_scaled, valid_eval, in_episode_eval)
    stats: SequenceTrainStats = train_sequence_model(
        model=model.network,
        x_train=x_train_input,
        y_train=y_train,
        sample_weight_train=sample_weight_train,
        x_eval=x_eval_input,
        y_eval=y_eval,
        sample_weight_eval=sample_weight_eval,
        random_seed=int(model.random_seed),
        batch_size=int(model.batch_size),
        max_epochs=int(model.max_epochs),
        early_stopping_patience=int(model.early_stopping_patience),
        learning_rate=float(model.sequence_learning_rate),
        weight_decay=float(model.weight_decay),
    )
    model.scaler_mean = scaler_mean
    model.scaler_std = scaler_std
    model.train_stats = stats.to_dict()
    model.training_history = list(stats.training_history)
    return model


def predict_detector_scores(
    model: SequenceDetector,
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    sequence_store: DetectorSequenceStore | None = None,
) -> pd.DataFrame:
    _require_columns(df, [*DETECTOR_IDENTITY_COLUMNS, "decision_row_id"], "df")
    if df.empty:
        return pd.DataFrame(columns=[*DETECTOR_IDENTITY_COLUMNS, "p_good"])
    if model.scaler_mean is None or model.scaler_std is None:
        raise ValueError("model is not fitted: scaler stats are missing")
    store = _resolve_sequence_store(model, sequence_store)
    x_raw, valid_mask, in_episode_mask = extract_sequences_for_rows(
        store, df["decision_row_id"].astype(str)
    )
    x_scaled = _transform_with_scaler(x_raw, valid_mask, model.scaler_mean, model.scaler_std)
    x_input = _stack_model_inputs(x_scaled, valid_mask, in_episode_mask)
    scores = predict_sequence_model_proba(model.network, x_input).astype(float)
    out = df.loc[:, list(DETECTOR_IDENTITY_COLUMNS)].copy()
    out["p_good"] = scores
    return out


def build_detector_feature_importance_table(
    model: Any, feature_columns: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    features = list(feature_columns)
    if not features:
        return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])
    if not hasattr(model, "get_feature_importance"):
        return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])
    raw_values = model.get_feature_importance(type="FeatureImportance")
    if len(raw_values) != len(features):
        raise ValueError(
            "feature importance length mismatch: "
            f"expected={len(features)} actual={len(raw_values)}"
        )
    table = pd.DataFrame(
        {
            "feature": features,
            "importance_raw": pd.to_numeric(pd.Series(raw_values), errors="coerce").fillna(0.0),
        }
    )
    table["importance_norm"] = _normalize_importance(table["importance_raw"])
    return table


def summarize_detector_oof_importance(
    oof_importance_df: pd.DataFrame, top_k: int = 20
) -> pd.DataFrame:
    required = ("fold_id", "feature", "importance_norm")
    _require_columns(oof_importance_df, list(required), "oof_importance_df")
    if oof_importance_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "top20_hit_rate",
                "top20_hits",
                "folds_total",
            ]
        )
    frame = oof_importance_df.copy()
    frame["importance_norm"] = pd.to_numeric(
        frame["importance_norm"], errors="coerce"
    ).fillna(0.0)
    top_ranked = (
        frame.sort_values(
            ["fold_id", "importance_norm", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("fold_id", sort=False, as_index=False)
        .head(int(top_k))
        .copy()
    )
    top_ranked["hit"] = 1
    folds_total = int(frame["fold_id"].nunique())
    stats = (
        frame.groupby("feature", as_index=False)["importance_norm"]
        .mean()
        .rename(columns={"importance_norm": "mean_importance"})
    )
    hits = (
        top_ranked.groupby("feature", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "top20_hits"})
    )
    out = stats.merge(hits, on="feature", how="left")
    out["top20_hits"] = out["top20_hits"].fillna(0).astype(int)
    out["folds_total"] = folds_total
    out["top20_hit_rate"] = (
        out["top20_hits"].astype(float) / float(max(folds_total, 1))
    )
    out = out.sort_values(
        ["mean_importance", "top20_hit_rate", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out.loc[
        :, ["feature", "mean_importance", "top20_hit_rate", "top20_hits", "folds_total"]
    ].copy()


def build_sequence_permutation_importance_table(
    model: SequenceDetector,
    eval_df: pd.DataFrame,
    target_column: str,
    sequence_store: DetectorSequenceStore | None = None,
) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])
    if model.scaler_mean is None or model.scaler_std is None:
        raise ValueError("model is not fitted: scaler stats are missing")
    _require_columns(eval_df, ["decision_row_id", target_column], "eval_df")
    store = _resolve_sequence_store(model, sequence_store)
    feature_columns = list(model.feature_columns)
    if not feature_columns:
        return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])
    x_eval_raw, valid_eval, in_episode_eval = extract_sequences_for_rows(
        store, eval_df["decision_row_id"].astype(str)
    )
    y_eval = (
        pd.to_numeric(eval_df[target_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if len(y_eval) == 0:
        return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])
    x_eval_scaled = _transform_with_scaler(
        x_eval_raw, valid_eval, model.scaler_mean, model.scaler_std
    )
    x_eval_input = _stack_model_inputs(x_eval_scaled, valid_eval, in_episode_eval)
    baseline_pred = predict_sequence_model_proba(model.network, x_eval_input).astype(float)
    baseline_loss = _binary_logloss(y_eval.astype(float), baseline_pred)
    rng = np.random.default_rng(int(model.random_seed))
    rows: list[dict[str, float | str]] = []
    for feature_idx, feature_name in enumerate(feature_columns):
        shuffled = x_eval_scaled.copy()
        perm = rng.permutation(shuffled.shape[0])
        shuffled[:, :, feature_idx] = shuffled[perm, :, feature_idx]
        shuffled_input = _stack_model_inputs(shuffled, valid_eval, in_episode_eval)
        perm_pred = predict_sequence_model_proba(model.network, shuffled_input).astype(float)
        perm_loss = _binary_logloss(y_eval.astype(float), perm_pred)
        rows.append(
            {
                "feature": str(feature_name),
                "importance_raw": float(perm_loss - baseline_loss),
            }
        )
    out = pd.DataFrame(rows, columns=["feature", "importance_raw"])
    out["importance_norm"] = _normalize_importance(out["importance_raw"])
    return out


def _normalize_importance(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(clipped.sum())
    if total > 0.0:
        return clipped / total
    if len(clipped) == 0:
        return clipped
    return pd.Series([1.0 / float(len(clipped))] * len(clipped), index=clipped.index)


def _resolve_sequence_store(
    model: SequenceDetector, sequence_store: DetectorSequenceStore | None
) -> DetectorSequenceStore:
    store = sequence_store if sequence_store is not None else model.sequence_store
    if store is None:
        raise ValueError("sequence_store is required for sequence detector")
    return store


def _fit_scaler(x: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_count = int(x.shape[2]) if x.ndim == 3 else 0
    mean = np.zeros(feature_count, dtype=np.float32)
    std = np.ones(feature_count, dtype=np.float32)
    for feature_idx in range(feature_count):
        values = x[:, :, feature_idx]
        finite_mask = valid_mask & np.isfinite(values)
        valid_values = values[finite_mask]
        if valid_values.size == 0:
            continue
        m = float(np.mean(valid_values))
        s = float(np.std(valid_values))
        mean[feature_idx] = m
        std[feature_idx] = s if s > 1e-6 else 1.0
    return mean, std


def _transform_with_scaler(
    x: np.ndarray, valid_mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    transformed = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    transformed = np.nan_to_num(
        transformed, nan=0.0, posinf=0.0, neginf=0.0
    )
    transformed = transformed.astype(np.float32, copy=False)
    transformed[~valid_mask] = 0.0
    return transformed


def _stack_model_inputs(
    x_scaled: np.ndarray, valid_mask: np.ndarray, in_episode_mask: np.ndarray
) -> np.ndarray:
    valid = valid_mask.astype(np.float32)[..., np.newaxis]
    in_episode = in_episode_mask.astype(np.float32)[..., np.newaxis]
    return np.concatenate([x_scaled, valid, in_episode], axis=2).astype(np.float32, copy=False)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _binary_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape[0] == 0:
        return 0.0
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))
