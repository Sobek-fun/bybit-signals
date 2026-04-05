from __future__ import annotations

import numpy as np
import pandas as pd


_PAIR_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "better_decision_row_id",
    "worse_decision_row_id",
    "pair_type",
    "pair_weight",
)


def build_detector_ranking_pairs(
    frame: pd.DataFrame,
    timeout_pair_weight: float,
    max_ranking_pairs_per_episode: int,
) -> pd.DataFrame:
    _require_columns(
        frame,
        [
            "episode_id",
            "decision_row_id",
            "episode_age_bars",
            "row_trade_outcome",
            "target_good_short_now",
            "trainable_row",
        ],
        "frame",
    )
    timeout_pair_weight = float(timeout_pair_weight)
    if timeout_pair_weight <= 0.0:
        raise ValueError("timeout_pair_weight must be positive")
    max_pairs = int(max_ranking_pairs_per_episode)
    if max_pairs <= 0:
        raise ValueError("max_ranking_pairs_per_episode must be positive")
    if frame.empty:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))
    local = frame.copy()
    local["trainable_row"] = local["trainable_row"].astype(bool)
    local = local[local["trainable_row"]].copy()
    if local.empty:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))
    local["episode_id"] = local["episode_id"].astype(str)
    local["decision_row_id"] = local["decision_row_id"].astype(str)
    local["episode_age_bars"] = pd.to_numeric(local["episode_age_bars"], errors="coerce")
    local["row_trade_outcome"] = (
        local["row_trade_outcome"].astype(str).str.strip().str.lower()
    )
    local = local[local["episode_age_bars"].notna()].copy()
    local = local[local["row_trade_outcome"].isin(["tp", "timeout", "sl"])].copy()
    if local.empty:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))
    local = local.sort_values(
        ["episode_id", "episode_age_bars", "decision_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for episode_id, group in local.groupby("episode_id", sort=False):
        tp = group[group["row_trade_outcome"] == "tp"].copy()
        timeout = group[group["row_trade_outcome"] == "timeout"].copy()
        sl = group[group["row_trade_outcome"] == "sl"].copy()
        episode_pairs: list[dict[str, object]] = []
        if not tp.empty and not sl.empty:
            episode_pairs.extend(
                _build_local_pairs(tp, sl, str(episode_id), "tp_vs_sl", 1.0)
            )
        if not tp.empty and not timeout.empty:
            episode_pairs.extend(
                _build_local_pairs(
                    tp,
                    timeout,
                    str(episode_id),
                    "tp_vs_timeout",
                    timeout_pair_weight,
                )
            )
        if not timeout.empty and not sl.empty:
            episode_pairs.extend(
                _build_local_pairs(
                    timeout,
                    sl,
                    str(episode_id),
                    "timeout_vs_sl",
                    timeout_pair_weight,
                )
            )
        if not episode_pairs:
            continue
        episode_pairs_df = pd.DataFrame(episode_pairs)
        episode_pairs_df["age_distance"] = pd.to_numeric(
            episode_pairs_df["age_distance"], errors="coerce"
        ).fillna(np.inf)
        episode_pairs_df = episode_pairs_df.sort_values(
            ["age_distance", "pair_type", "better_decision_row_id", "worse_decision_row_id"],
            kind="mergesort",
        ).head(max_pairs)
        rows.extend(
            episode_pairs_df.loc[:, list(_PAIR_COLUMNS)].to_dict(orient="records")
        )
    if not rows:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))
    out = pd.DataFrame(rows, columns=list(_PAIR_COLUMNS))
    return out.reset_index(drop=True)


def build_hard_negative_row_weights(
    frame: pd.DataFrame,
    hard_negative_weight_multiplier: float,
    hard_negative_max_age_distance: int,
    timeout_hard_negative_weight_multiplier: float = 1.5,
    return_hard_negative_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    _require_columns(
        frame,
        [
            "episode_id",
            "episode_age_bars",
            "row_trade_outcome",
            "target_good_short_now",
            "trainable_row",
        ],
        "frame",
    )
    base = (
        pd.to_numeric(frame["target_row_weight"], errors="coerce").fillna(1.0)
        if "target_row_weight" in frame.columns
        else pd.Series(np.ones(len(frame), dtype=np.float32), index=frame.index)
    )
    if frame.empty:
        weights_empty = np.asarray(base, dtype=np.float32)
        if return_hard_negative_mask:
            return weights_empty, np.zeros(len(frame), dtype=np.bool_)
        return weights_empty
    sl_multiplier = float(hard_negative_weight_multiplier)
    timeout_multiplier = float(timeout_hard_negative_weight_multiplier)
    if sl_multiplier <= 0.0:
        raise ValueError("hard_negative_weight_multiplier must be positive")
    if timeout_multiplier <= 0.0:
        raise ValueError("timeout_hard_negative_weight_multiplier must be positive")
    max_age_distance = int(hard_negative_max_age_distance)
    if max_age_distance < 0:
        raise ValueError("hard_negative_max_age_distance must be non-negative")
    local = frame.copy()
    local["_idx"] = np.arange(len(local), dtype=np.int64)
    local["trainable_row"] = local["trainable_row"].astype(bool)
    local["episode_id"] = local["episode_id"].astype(str)
    local["episode_age_bars"] = pd.to_numeric(local["episode_age_bars"], errors="coerce")
    local["row_trade_outcome"] = (
        local["row_trade_outcome"].astype(str).str.strip().str.lower()
    )
    local["target_good_short_now"] = (
        pd.to_numeric(local["target_good_short_now"], errors="coerce").fillna(0.0).astype(int)
    )
    trainable = local[
        local["trainable_row"] & local["episode_age_bars"].notna()
    ].copy()
    if trainable.empty:
        weights = np.asarray(base, dtype=np.float32)
        if return_hard_negative_mask:
            return weights, np.zeros(len(frame), dtype=np.bool_)
        return weights
    hard_negative_mask = np.zeros(len(frame), dtype=np.bool_)
    for _, group in trainable.groupby("episode_id", sort=False):
        tp_ages = (
            group.loc[group["row_trade_outcome"] == "tp", "episode_age_bars"]
            .dropna()
            .to_numpy(dtype=float)
        )
        if tp_ages.size == 0:
            continue
        negatives = group[group["row_trade_outcome"].isin(["sl", "timeout"])].copy()
        if negatives.empty:
            continue
        neg_ages = negatives["episode_age_bars"].to_numpy(dtype=float)
        nearest_dist = np.min(np.abs(neg_ages.reshape(-1, 1) - tp_ages.reshape(1, -1)), axis=1)
        is_hard = nearest_dist <= float(max_age_distance)
        if not np.any(is_hard):
            continue
        idxs = negatives.loc[is_hard, "_idx"].to_numpy(dtype=int)
        hard_negative_mask[idxs] = True
    weights = np.asarray(base, dtype=np.float32).copy()
    outcomes = local["row_trade_outcome"].to_numpy(dtype=object)
    sl_hard = hard_negative_mask & (outcomes == "sl")
    timeout_hard = hard_negative_mask & (outcomes == "timeout")
    weights[sl_hard] *= np.float32(sl_multiplier)
    weights[timeout_hard] *= np.float32(timeout_multiplier)
    if return_hard_negative_mask:
        return weights.astype(np.float32, copy=False), hard_negative_mask
    return weights.astype(np.float32, copy=False)


def _build_local_pairs(
    better: pd.DataFrame,
    worse: pd.DataFrame,
    episode_id: str,
    pair_type: str,
    pair_weight: float,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for better_row in better.itertuples(index=False):
        better_age = float(getattr(better_row, "episode_age_bars"))
        better_id = str(getattr(better_row, "decision_row_id"))
        local_worse = worse.copy()
        local_worse["age_distance"] = (
            pd.to_numeric(local_worse["episode_age_bars"], errors="coerce") - better_age
        ).abs()
        local_worse = local_worse.sort_values(
            ["age_distance", "decision_row_id"], kind="mergesort"
        )
        for worse_row in local_worse.itertuples(index=False):
            out.append(
                {
                    "episode_id": episode_id,
                    "better_decision_row_id": better_id,
                    "worse_decision_row_id": str(getattr(worse_row, "decision_row_id")),
                    "pair_type": pair_type,
                    "pair_weight": float(pair_weight),
                    "age_distance": float(getattr(worse_row, "age_distance")),
                }
            )
    return out


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
