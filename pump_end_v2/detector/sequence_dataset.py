from dataclasses import dataclass

import numpy as np
import pandas as pd

from pump_end_v2.features.manifest import (
    DETECTOR_IDENTITY_COLUMNS,
    DETECTOR_SEQUENCE_FEATURE_COLUMNS,
)

SEQUENCE_LOOKBACK_BARS = 16
_BAR_NS = int(pd.Timedelta(minutes=15).value)

_META_COLUMNS: tuple[str, ...] = (
    *DETECTOR_IDENTITY_COLUMNS,
    "dataset_split",
    "trainable_row",
    "target_good_short_now",
    "target_reason",
)

_TOKEN_TIME_COLUMN = "open_time"
_REFERENCE_TIME_COLUMN = "open_time"
_BREADTH_TIME_COLUMN = "open_time"
_EPISODE_TIME_COLUMN = "context_bar_open_time"
_EPISODE_ID_COLUMN = "episode_id"
_SYMBOL_COLUMN = "symbol"


@dataclass(slots=True)
class DetectorSequenceStore:
    meta_df: pd.DataFrame
    x: np.ndarray
    valid_mask: np.ndarray
    in_episode_mask: np.ndarray
    row_index_by_decision_row_id: dict[str, int]
    feature_columns: tuple[str, ...]
    lookback_bars: int


def build_detector_sequence_store(
    detector_dataset_df: pd.DataFrame,
    token_state_tradable_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    episode_state_df: pd.DataFrame,
    feature_columns: tuple[str, ...] = DETECTOR_SEQUENCE_FEATURE_COLUMNS,
    lookback_bars: int = SEQUENCE_LOOKBACK_BARS,
) -> DetectorSequenceStore:
    if int(lookback_bars) != SEQUENCE_LOOKBACK_BARS:
        raise ValueError(f"lookback_bars must be {SEQUENCE_LOOKBACK_BARS}")
    _require_columns(detector_dataset_df, _META_COLUMNS, "detector_dataset_df")
    feature_columns = tuple(feature_columns)
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    meta_df = detector_dataset_df.loc[:, list(_META_COLUMNS)].copy()
    meta_df["context_bar_open_time"] = pd.to_datetime(
        meta_df["context_bar_open_time"], utc=True, errors="raise"
    )
    meta_df["decision_time"] = pd.to_datetime(
        meta_df["decision_time"], utc=True, errors="raise"
    )
    meta_df["entry_bar_open_time"] = pd.to_datetime(
        meta_df["entry_bar_open_time"], utc=True, errors="raise"
    )
    token_feature_columns = _resolve_source_columns(
        feature_columns,
        tuple(
            column
            for column in token_state_tradable_df.columns
            if column not in ("symbol", "open_time")
        ),
    )
    reference_feature_columns = _resolve_source_columns(
        feature_columns,
        tuple(
            column
            for column in reference_state_df.columns
            if column != _REFERENCE_TIME_COLUMN
        ),
    )
    breadth_feature_columns = _resolve_source_columns(
        feature_columns,
        tuple(
            column
            for column in breadth_state_df.columns
            if column != _BREADTH_TIME_COLUMN
        ),
    )
    episode_feature_columns = _resolve_source_columns(
        feature_columns,
        tuple(
            column
            for column in episode_state_df.columns
            if column
            not in ("decision_row_id", "episode_id", "symbol", "context_bar_open_time")
        ),
    )
    covered = (
        set(token_feature_columns)
        | set(reference_feature_columns)
        | set(breadth_feature_columns)
        | set(episode_feature_columns)
    )
    missing = [feature for feature in feature_columns if feature not in covered]
    if missing:
        raise ValueError(f"unmapped sequence features: {missing}")
    feature_pos = {feature: idx for idx, feature in enumerate(feature_columns)}
    token_pos = {feature: idx for idx, feature in enumerate(token_feature_columns)}
    reference_pos = {
        feature: idx for idx, feature in enumerate(reference_feature_columns)
    }
    breadth_pos = {feature: idx for idx, feature in enumerate(breadth_feature_columns)}
    episode_pos = {feature: idx for idx, feature in enumerate(episode_feature_columns)}
    token_lookup = _build_token_lookup(token_state_tradable_df, token_feature_columns)
    reference_lookup = _build_time_lookup(
        reference_state_df, _REFERENCE_TIME_COLUMN, reference_feature_columns
    )
    breadth_lookup = _build_time_lookup(
        breadth_state_df, _BREADTH_TIME_COLUMN, breadth_feature_columns
    )
    episode_lookup = _build_episode_lookup(episode_state_df, episode_feature_columns)
    episode_open_ts = _build_episode_open_times(meta_df)
    rows_total = len(meta_df)
    feature_total = len(feature_columns)
    x = np.zeros((rows_total, lookback_bars, feature_total), dtype=np.float32)
    valid_mask = np.zeros((rows_total, lookback_bars), dtype=np.bool_)
    in_episode_mask = np.zeros((rows_total, lookback_bars), dtype=np.bool_)
    for row_idx, row in enumerate(meta_df.itertuples(index=False)):
        symbol = str(getattr(row, _SYMBOL_COLUMN))
        episode_id = str(getattr(row, _EPISODE_ID_COLUMN))
        context_ts = pd.Timestamp(getattr(row, "context_bar_open_time")).value
        open_ts = int(episode_open_ts.get(episode_id, context_ts + 1))
        for step in range(lookback_bars):
            ts = context_ts - (lookback_bars - 1 - step) * _BAR_NS
            token_values = token_lookup.get((symbol, ts))
            if token_values is not None:
                valid_mask[row_idx, step] = True
            reference_values = reference_lookup.get(ts)
            breadth_values = breadth_lookup.get(ts)
            is_in_episode = open_ts <= ts <= context_ts
            if is_in_episode:
                in_episode_mask[row_idx, step] = True
            episode_values = episode_lookup.get((episode_id, ts)) if is_in_episode else None
            for feature, pos in feature_pos.items():
                value = 0.0
                if feature in token_pos and token_values is not None:
                    value = float(token_values[token_pos[feature]])
                elif feature in reference_pos and reference_values is not None:
                    value = float(reference_values[reference_pos[feature]])
                elif feature in breadth_pos and breadth_values is not None:
                    value = float(breadth_values[breadth_pos[feature]])
                elif (
                    feature in episode_pos
                    and episode_values is not None
                    and is_in_episode
                ):
                    value = float(episode_values[episode_pos[feature]])
                x[row_idx, step, pos] = value
    invalid_mask = ~np.isfinite(x)
    if invalid_mask.any():
        invalid_positions = np.argwhere(invalid_mask)
        invalid_count = int(invalid_positions.shape[0])
        sample_positions = [
            (int(pos[0]), int(pos[1]), int(pos[2]))
            for pos in invalid_positions[:10]
        ]
        raise ValueError(
            "sequence store contains non-finite values: "
            f"count={invalid_count} sample_positions={sample_positions}"
        )
    row_index = {
        str(row.decision_row_id): idx
        for idx, row in enumerate(meta_df.loc[:, ["decision_row_id"]].itertuples(index=False))
    }
    return DetectorSequenceStore(
        meta_df=meta_df.reset_index(drop=True),
        x=x,
        valid_mask=valid_mask,
        in_episode_mask=in_episode_mask,
        row_index_by_decision_row_id=row_index,
        feature_columns=feature_columns,
        lookback_bars=lookback_bars,
    )


def extract_sequences_for_rows(
    sequence_store: DetectorSequenceStore, decision_row_ids: pd.Series | list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = pd.Series(decision_row_ids).astype(str).tolist()
    if not ids:
        f = len(sequence_store.feature_columns)
        t = int(sequence_store.lookback_bars)
        return (
            np.zeros((0, t, f), dtype=np.float32),
            np.zeros((0, t), dtype=np.bool_),
            np.zeros((0, t), dtype=np.bool_),
        )
    indices: list[int] = []
    missing: list[str] = []
    for decision_row_id in ids:
        row_idx = sequence_store.row_index_by_decision_row_id.get(decision_row_id)
        if row_idx is None:
            missing.append(decision_row_id)
            continue
        indices.append(int(row_idx))
    if missing:
        raise ValueError(
            f"sequence rows missing decision_row_id count={len(missing)} sample={missing[:5]}"
        )
    x = sequence_store.x[indices].astype(np.float32, copy=False)
    valid = sequence_store.valid_mask[indices].astype(np.bool_, copy=False)
    in_episode = sequence_store.in_episode_mask[indices].astype(np.bool_, copy=False)
    return x, valid, in_episode


def _resolve_source_columns(
    feature_columns: tuple[str, ...], source_columns: tuple[str, ...]
) -> tuple[str, ...]:
    source_set = set(source_columns)
    return tuple(feature for feature in feature_columns if feature in source_set)


def _build_token_lookup(
    token_state_df: pd.DataFrame, feature_columns: tuple[str, ...]
) -> dict[tuple[str, int], np.ndarray]:
    required = ("symbol", "open_time", *feature_columns)
    _require_columns(token_state_df, required, "token_state_tradable_df")
    frame = token_state_df.loc[:, list(required)].copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="raise")
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        symbol = str(row.symbol)
        ts = pd.Timestamp(row.open_time).value
        numeric_values = pd.to_numeric(
            pd.Series(row[2:]), errors="coerce"
        ).fillna(0.0)
        values = numeric_values.to_numpy(dtype=np.float32, copy=False)
        values = np.where(np.isfinite(values), values, 0.0).astype(
            np.float32, copy=False
        )
        out[(symbol, ts)] = values
    return out


def _build_time_lookup(
    source_df: pd.DataFrame, time_column: str, feature_columns: tuple[str, ...]
) -> dict[int, np.ndarray]:
    required = (time_column, *feature_columns)
    _require_columns(source_df, required, "time_feature_df")
    frame = source_df.loc[:, list(required)].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    out: dict[int, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        ts = pd.Timestamp(row[0]).value
        numeric_values = pd.to_numeric(
            pd.Series(row[1:]), errors="coerce"
        ).fillna(0.0)
        values = numeric_values.to_numpy(dtype=np.float32, copy=False)
        values = np.where(np.isfinite(values), values, 0.0).astype(
            np.float32, copy=False
        )
        out[ts] = values
    return out


def _build_episode_lookup(
    episode_state_df: pd.DataFrame, feature_columns: tuple[str, ...]
) -> dict[tuple[str, int], np.ndarray]:
    required = ("episode_id", "context_bar_open_time", *feature_columns)
    _require_columns(episode_state_df, required, "episode_state_df")
    frame = episode_state_df.loc[:, list(required)].copy()
    frame["context_bar_open_time"] = pd.to_datetime(
        frame["context_bar_open_time"], utc=True, errors="raise"
    )
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.episode_id), pd.Timestamp(row.context_bar_open_time).value)
        numeric_values = pd.to_numeric(
            pd.Series(row[2:]), errors="coerce"
        ).fillna(0.0)
        values = numeric_values.to_numpy(dtype=np.float32, copy=False)
        values = np.where(np.isfinite(values), values, 0.0).astype(
            np.float32, copy=False
        )
        out[key] = values
    return out


def _build_episode_open_times(meta_df: pd.DataFrame) -> dict[str, int]:
    grouped = (
        meta_df.loc[:, ["episode_id", "context_bar_open_time"]]
        .groupby("episode_id", as_index=False)["context_bar_open_time"]
        .min()
    )
    return {
        str(row.episode_id): pd.Timestamp(row.context_bar_open_time).value
        for row in grouped.itertuples(index=False)
    }


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
