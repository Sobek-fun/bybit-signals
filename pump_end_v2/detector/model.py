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
    predict_sequence_model_outputs,
    train_sequence_model,
)
from pump_end_v2.detector.ranking import (
    build_detector_ranking_pairs,
    build_hard_negative_row_weights,
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
    ranking_lambda: float
    hard_negative_weight_multiplier: float
    hard_negative_max_age_distance: int
    max_ranking_pairs_per_episode: int
    timeout_pair_weight: float
    outcome_aux_lambda: float
    main_target_mode: str
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
        ranking_lambda=float(model_config.ranking_lambda),
        hard_negative_weight_multiplier=float(model_config.hard_negative_weight_multiplier),
        hard_negative_max_age_distance=int(model_config.hard_negative_max_age_distance),
        max_ranking_pairs_per_episode=int(model_config.max_ranking_pairs_per_episode),
        timeout_pair_weight=float(model_config.timeout_pair_weight),
        outcome_aux_lambda=float(model_config.outcome_aux_lambda),
        main_target_mode=str(model_config.main_target_mode),
    )


def fit_detector_model(
    model: SequenceDetector,
    train_df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    eval_df: pd.DataFrame | None = None,
    sequence_store: DetectorSequenceStore | None = None,
) -> SequenceDetector:
    _require_columns(
        train_df,
        [
            "decision_row_id",
            "episode_id",
            "trainable_row",
            "is_ideal_entry",
            "target_reason",
        ],
        "train_df",
    )
    store = _resolve_sequence_store(model, sequence_store)
    model.sequence_store = store
    train_fit = _prepare_fit_rows(train_df)
    if train_fit.empty:
        raise ValueError("fit_detector_model received empty train rows")
    x_train_raw, valid_train, in_episode_train, readout_index_train = (
        extract_sequences_for_rows(store, train_fit["decision_row_id"].astype(str))
    )
    scaler_mean, scaler_std = _fit_scaler(x_train_raw, valid_train)
    x_train_scaled = _transform_with_scaler(x_train_raw, valid_train, scaler_mean, scaler_std)
    x_train_input = _stack_model_inputs(x_train_scaled, valid_train, in_episode_train)
    train_episode_ids = train_fit["episode_id"].astype(str).to_numpy(dtype=object)
    y_train = (
        pd.to_numeric(train_fit[target_column], errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy(copy=False)
    )
    sample_weight_train, hard_negative_train_mask = build_hard_negative_row_weights(
        train_fit,
        hard_negative_weight_multiplier=float(model.hard_negative_weight_multiplier),
        hard_negative_max_age_distance=int(model.hard_negative_max_age_distance),
        return_hard_negative_mask=True,
    )
    ranking_pairs_train_df = build_detector_ranking_pairs(
        train_fit,
        timeout_pair_weight=float(model.timeout_pair_weight),
        max_ranking_pairs_per_episode=int(model.max_ranking_pairs_per_episode),
    )
    if model.main_target_mode == "tp_vs_sl_only":
        train_target_reason = train_fit["target_reason"].astype(str).str.strip().str.lower()
        timeout_train_mask = train_target_reason.eq("timeout").to_numpy(copy=False)
        sample_weight_train = sample_weight_train.astype(np.float32, copy=True)
        sample_weight_train[timeout_train_mask] = 0.0
        if not ranking_pairs_train_df.empty:
            ranking_pairs_train_df = ranking_pairs_train_df[
                ranking_pairs_train_df["pair_type"].astype(str).eq("tp_vs_sl")
            ].copy()
    ranking_pairs_train = _prepare_pair_index_data(
        ranking_pairs_train_df, train_fit["decision_row_id"].astype(str)
    )
    train_outcome_targets = _encode_outcome_targets(train_fit["target_reason"])
    train_outcome_weights = _resolve_outcome_weights(train_fit)
    x_eval_input: np.ndarray | None = None
    y_eval: np.ndarray | None = None
    sample_weight_eval: np.ndarray | None = None
    eval_episode_ids: np.ndarray | None = None
    eval_readout_index: np.ndarray | None = None
    ranking_pairs_eval: dict[str, np.ndarray] | None = None
    hard_negative_eval_total = 0
    eval_outcome_targets: np.ndarray | None = None
    eval_outcome_weights: np.ndarray | None = None
    if eval_df is not None and len(eval_df) > 0:
        _require_columns(
            eval_df,
            [
                "decision_row_id",
                "episode_id",
                "trainable_row",
                "is_ideal_entry",
                "target_reason",
            ],
            "eval_df",
        )
        eval_fit = _prepare_fit_rows(eval_df)
        if not eval_fit.empty:
            x_eval_raw, valid_eval, in_episode_eval, eval_readout_index = (
                extract_sequences_for_rows(store, eval_fit["decision_row_id"].astype(str))
            )
            eval_outcome_targets = _encode_outcome_targets(eval_fit["target_reason"])
            eval_outcome_weights = _resolve_outcome_weights(eval_fit)
            eval_episode_ids = eval_fit["episode_id"].astype(str).to_numpy(dtype=object)
            y_eval = (
                pd.to_numeric(eval_fit[target_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
                .to_numpy(copy=False)
            )
            sample_weight_eval, hard_negative_eval_mask = build_hard_negative_row_weights(
                eval_fit,
                hard_negative_weight_multiplier=float(model.hard_negative_weight_multiplier),
                hard_negative_max_age_distance=int(model.hard_negative_max_age_distance),
                return_hard_negative_mask=True,
            )
            hard_negative_eval_total = int(np.sum(hard_negative_eval_mask))
            ranking_pairs_eval_df = build_detector_ranking_pairs(
                eval_fit,
                timeout_pair_weight=float(model.timeout_pair_weight),
                max_ranking_pairs_per_episode=int(model.max_ranking_pairs_per_episode),
            )
            if model.main_target_mode == "tp_vs_sl_only":
                eval_target_reason = (
                    eval_fit["target_reason"].astype(str).str.strip().str.lower()
                )
                timeout_eval_mask = eval_target_reason.eq("timeout").to_numpy(copy=False)
                sample_weight_eval = sample_weight_eval.astype(np.float32, copy=True)
                sample_weight_eval[timeout_eval_mask] = 0.0
                if not ranking_pairs_eval_df.empty:
                    ranking_pairs_eval_df = ranking_pairs_eval_df[
                        ranking_pairs_eval_df["pair_type"].astype(str).eq("tp_vs_sl")
                    ].copy()
            ranking_pairs_eval = _prepare_pair_index_data(
                ranking_pairs_eval_df, eval_fit["decision_row_id"].astype(str)
            )
            x_eval_scaled = _transform_with_scaler(
                x_eval_raw, valid_eval, scaler_mean, scaler_std
            )
            x_eval_input = _stack_model_inputs(x_eval_scaled, valid_eval, in_episode_eval)
    hard_negative_train_total = int(np.sum(hard_negative_train_mask))
    stats: SequenceTrainStats = train_sequence_model(
        model=model.network,
        x_train=x_train_input,
        y_train=y_train,
        sample_weight_train=sample_weight_train,
        train_episode_ids=train_episode_ids,
        train_decision_row_ids=None,
        train_readout_index=readout_index_train,
        train_episode_target_row_id_map=None,
        train_outcome_targets=train_outcome_targets,
        train_outcome_weights=train_outcome_weights,
        x_eval=x_eval_input,
        y_eval=y_eval,
        sample_weight_eval=sample_weight_eval,
        eval_episode_ids=eval_episode_ids,
        eval_decision_row_ids=None,
        eval_readout_index=eval_readout_index,
        eval_episode_target_row_id_map=None,
        eval_outcome_targets=eval_outcome_targets,
        eval_outcome_weights=eval_outcome_weights,
        random_seed=int(model.random_seed),
        batch_size=int(model.batch_size),
        max_epochs=int(model.max_epochs),
        early_stopping_patience=int(model.early_stopping_patience),
        learning_rate=float(model.sequence_learning_rate),
        weight_decay=float(model.weight_decay),
        outcome_aux_lambda=float(model.outcome_aux_lambda),
        ranking_pairs_train=ranking_pairs_train,
        ranking_pairs_eval=ranking_pairs_eval,
        ranking_lambda=float(model.ranking_lambda),
        hard_negative_rows_train_total=hard_negative_train_total,
        hard_negative_rows_eval_total=hard_negative_eval_total,
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
        return pd.DataFrame(
            columns=[
                *DETECTOR_IDENTITY_COLUMNS,
                "p_good",
                "p_tp_row",
                "p_timeout_row",
                "p_sl_row",
            ]
        )
    if model.scaler_mean is None or model.scaler_std is None:
        raise ValueError("model is not fitted: scaler stats are missing")
    store = _resolve_sequence_store(model, sequence_store)
    x_raw, valid_mask, in_episode_mask, readout_index = extract_sequences_for_rows(
        store, df["decision_row_id"].astype(str)
    )
    x_scaled = _transform_with_scaler(x_raw, valid_mask, model.scaler_mean, model.scaler_std)
    x_input = _stack_model_inputs(x_scaled, valid_mask, in_episode_mask)
    outputs = predict_sequence_model_outputs(model.network, x_input, readout_index)
    out = df.loc[:, list(DETECTOR_IDENTITY_COLUMNS)].copy()
    out["p_good"] = outputs["p_good"].astype(float)
    out["p_tp_row"] = outputs["p_tp_row"].astype(float)
    out["p_timeout_row"] = outputs["p_timeout_row"].astype(float)
    out["p_sl_row"] = outputs["p_sl_row"].astype(float)
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
    return pd.DataFrame(columns=["feature", "importance_raw", "importance_norm"])


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


def _prepare_fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out[out["trainable_row"].astype(bool)].copy()
    if out.empty:
        return out
    sort_columns = ["episode_id", "decision_row_id"]
    if "episode_age_bars" in out.columns:
        sort_columns = ["episode_id", "episode_age_bars", "decision_row_id"]
    elif "context_bar_open_time" in out.columns:
        sort_columns = ["episode_id", "context_bar_open_time", "decision_row_id"]
    return out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _build_episode_target_row_id_map(frame: pd.DataFrame) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for episode_id, group in frame.groupby("episode_id", sort=False):
        g = group.copy()
        ideal = g[g["is_ideal_entry"].astype(bool)].copy()
        if ideal.empty:
            mapping[str(episode_id)] = None
            continue
        ideal = ideal.sort_values(
            ["context_bar_open_time", "decision_row_id"], kind="mergesort"
        )
        mapping[str(episode_id)] = str(ideal.iloc[0]["decision_row_id"])
    return mapping


def _encode_outcome_targets(target_reason: pd.Series) -> np.ndarray:
    reason = target_reason.astype(str).str.strip().str.lower()
    out = np.full(len(reason), -1, dtype=np.int64)
    out[reason.eq("tp").to_numpy(dtype=bool, copy=False)] = 0
    out[reason.eq("timeout").to_numpy(dtype=bool, copy=False)] = 1
    out[reason.eq("sl").to_numpy(dtype=bool, copy=False)] = 2
    return out


def _resolve_outcome_weights(frame: pd.DataFrame) -> np.ndarray:
    if "target_row_weight" not in frame.columns:
        return np.ones(len(frame), dtype=np.float32)
    weights = pd.to_numeric(frame["target_row_weight"], errors="coerce").fillna(1.0)
    weights = weights.clip(lower=0.0)
    return weights.to_numpy(dtype=np.float32, copy=False)


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


def _prepare_pair_index_data(
    ranking_pairs_df: pd.DataFrame, decision_row_ids: pd.Series
) -> dict[str, np.ndarray]:
    if ranking_pairs_df.empty:
        return {
            "better_idx": np.zeros(0, dtype=np.int64),
            "worse_idx": np.zeros(0, dtype=np.int64),
            "pair_weight": np.zeros(0, dtype=np.float32),
        }
    id_to_pos = {str(row_id): idx for idx, row_id in enumerate(decision_row_ids.tolist())}
    better_ids = ranking_pairs_df["better_decision_row_id"].astype(str).to_numpy(dtype=object)
    worse_ids = ranking_pairs_df["worse_decision_row_id"].astype(str).to_numpy(dtype=object)
    pair_weights = pd.to_numeric(ranking_pairs_df["pair_weight"], errors="coerce").fillna(0.0)
    better_idx: list[int] = []
    worse_idx: list[int] = []
    weights: list[float] = []
    for b_id, w_id, weight in zip(better_ids, worse_ids, pair_weights, strict=False):
        b_pos = id_to_pos.get(str(b_id))
        w_pos = id_to_pos.get(str(w_id))
        if b_pos is None or w_pos is None:
            continue
        if float(weight) <= 0.0:
            continue
        better_idx.append(int(b_pos))
        worse_idx.append(int(w_pos))
        weights.append(float(weight))
    return {
        "better_idx": np.asarray(better_idx, dtype=np.int64),
        "worse_idx": np.asarray(worse_idx, dtype=np.int64),
        "pair_weight": np.asarray(weights, dtype=np.float32),
    }
