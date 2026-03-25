from __future__ import annotations

from collections import deque

import pandas as pd

from pump_end_v2.logging import log_info

GATE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "score_source",
    "fold_id",
)

GATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "detector_p_good",
    "detector_peak_p_good_before_fire",
    "detector_p_good_drop_from_peak",
    "detector_episode_age_bars",
    "detector_distance_from_episode_high_pct",
    "token_close_ret_1",
    "token_close_ret_4",
    "token_close_ret_12",
    "token_intrabar_range_pct",
    "token_rolling_volatility_4",
    "token_rolling_volatility_12",
    "token_runup_pct",
    "token_volume_ratio",
    "token_pump_context_flag",
    "btc_close_ret_1",
    "btc_close_ret_4",
    "btc_close_ret_12",
    "btc_intrabar_range_pct",
    "btc_volume_ratio",
    "btc_pump_context_flag",
    "eth_close_ret_1",
    "eth_close_ret_4",
    "eth_close_ret_12",
    "eth_intrabar_range_pct",
    "eth_volume_ratio",
    "eth_pump_context_flag",
    "breadth_universe_size",
    "breadth_advancers_share",
    "breadth_mean_close_ret_1",
    "breadth_median_close_ret_1",
    "breadth_std_close_ret_1",
    "breadth_near_high_share",
    "breadth_pump_context_share",
    "breadth_volume_spike_share",
    "signal_flow_recent_signals_24h",
    "signal_flow_recent_same_symbol_signals_24h",
    "signal_flow_recent_mean_detector_p_good_24h",
)

_CANDIDATE_PROD_REQUIRED_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "p_good",
    "peak_p_good_before_fire",
    "p_good_drop_from_peak",
    "episode_age_bars",
    "distance_from_episode_high_pct",
)

def build_gate_feature_view(
    candidate_signals_df: pd.DataFrame,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(candidate_signals_df, _CANDIDATE_PROD_REQUIRED_COLUMNS, "candidate_signals_df")
    _require_columns(
        token_state_df,
        (
            "symbol",
            "open_time",
            "close_ret_1",
            "close_ret_4",
            "close_ret_12",
            "intrabar_range_pct",
            "rolling_volatility_4",
            "rolling_volatility_12",
            "runup_pct",
            "volume_ratio",
            "pump_context_flag",
        ),
        "token_state_df",
    )
    _require_columns(
        reference_state_df,
        (
            "open_time",
            "btc_close_ret_1",
            "btc_close_ret_4",
            "btc_close_ret_12",
            "btc_intrabar_range_pct",
            "btc_volume_ratio",
            "btc_pump_context_flag",
            "eth_close_ret_1",
            "eth_close_ret_4",
            "eth_close_ret_12",
            "eth_intrabar_range_pct",
            "eth_volume_ratio",
            "eth_pump_context_flag",
        ),
        "reference_state_df",
    )
    _require_columns(
        breadth_state_df,
        (
            "open_time",
            "breadth_universe_size",
            "breadth_advancers_share",
            "breadth_mean_close_ret_1",
            "breadth_median_close_ret_1",
            "breadth_std_close_ret_1",
            "breadth_near_high_share",
            "breadth_pump_context_share",
            "breadth_volume_spike_share",
        ),
        "breadth_state_df",
    )
    frame = candidate_signals_df.copy()
    frame["context_bar_open_time"] = pd.to_datetime(frame["context_bar_open_time"], utc=True, errors="raise")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    frame["entry_bar_open_time"] = pd.to_datetime(frame["entry_bar_open_time"], utc=True, errors="raise")
    if "score_source" not in frame.columns:
        frame["score_source"] = "unknown"
    if "fold_id" not in frame.columns:
        frame["fold_id"] = pd.NA
    token_part = token_state_df[
        [
            "symbol",
            "open_time",
            "close_ret_1",
            "close_ret_4",
            "close_ret_12",
            "intrabar_range_pct",
            "rolling_volatility_4",
            "rolling_volatility_12",
            "runup_pct",
            "volume_ratio",
            "pump_context_flag",
        ]
    ].rename(
        columns={
            "open_time": "context_bar_open_time",
            "close_ret_1": "token_close_ret_1",
            "close_ret_4": "token_close_ret_4",
            "close_ret_12": "token_close_ret_12",
            "intrabar_range_pct": "token_intrabar_range_pct",
            "rolling_volatility_4": "token_rolling_volatility_4",
            "rolling_volatility_12": "token_rolling_volatility_12",
            "runup_pct": "token_runup_pct",
            "volume_ratio": "token_volume_ratio",
            "pump_context_flag": "token_pump_context_flag",
        }
    )
    merged = _merge_required_rows(
        left_df=frame,
        right_df=token_part,
        on=["symbol", "context_bar_open_time"],
        source_name="token_state_df",
    )
    ref_part = reference_state_df.rename(columns={"open_time": "context_bar_open_time"})
    merged = _merge_required_rows(
        left_df=merged,
        right_df=ref_part,
        on=["context_bar_open_time"],
        source_name="reference_state_df",
    )
    breadth_part = breadth_state_df.rename(columns={"open_time": "context_bar_open_time"})
    merged = _merge_required_rows(
        left_df=merged,
        right_df=breadth_part,
        on=["context_bar_open_time"],
        source_name="breadth_state_df",
    )
    merged["detector_p_good"] = pd.to_numeric(merged["p_good"], errors="coerce")
    merged["detector_peak_p_good_before_fire"] = pd.to_numeric(merged["peak_p_good_before_fire"], errors="coerce")
    merged["detector_p_good_drop_from_peak"] = pd.to_numeric(merged["p_good_drop_from_peak"], errors="coerce")
    merged["detector_episode_age_bars"] = pd.to_numeric(merged["episode_age_bars"], errors="coerce")
    merged["detector_distance_from_episode_high_pct"] = pd.to_numeric(
        merged["distance_from_episode_high_pct"], errors="coerce"
    )
    merged = _append_signal_flow_features(merged)
    out = merged.loc[:, [*GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS]].copy()
    log_info(
        "GATE",
        f"gate feature view build done rows_total={len(out)} feature_cols_total={len(GATE_FEATURE_COLUMNS)}",
    )
    return out


def _append_signal_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["_stream_group_key"] = frame.apply(_stream_group_key, axis=1)
    frame = frame.sort_values(
        ["_stream_group_key", "context_bar_open_time", "decision_time", "signal_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    recent_counts = [0] * len(frame)
    same_symbol_counts = [0] * len(frame)
    recent_mean_scores = [0.0] * len(frame)
    window = pd.Timedelta(hours=24)
    for _, idx in frame.groupby("_stream_group_key", sort=False, dropna=False).groups.items():
        history: deque[tuple[pd.Timestamp, str, float]] = deque()
        score_sum = 0.0
        symbol_counts: dict[str, int] = {}
        for row_idx in idx:
            row = frame.iloc[row_idx]
            now = pd.Timestamp(row["context_bar_open_time"])
            while history and history[0][0] < now - window:
                old_time, old_symbol, old_score = history.popleft()
                score_sum -= old_score
                current_count = symbol_counts.get(old_symbol, 0)
                if current_count <= 1:
                    symbol_counts.pop(old_symbol, None)
                else:
                    symbol_counts[old_symbol] = current_count - 1
            total_prev = len(history)
            recent_counts[row_idx] = total_prev
            current_symbol = str(row["symbol"])
            same_symbol_counts[row_idx] = symbol_counts.get(current_symbol, 0)
            recent_mean_scores[row_idx] = float(score_sum / total_prev) if total_prev > 0 else 0.0
            p_good_value = float(pd.to_numeric(row["detector_p_good"], errors="coerce"))
            if pd.isna(p_good_value):
                p_good_value = 0.0
            history.append((now, current_symbol, p_good_value))
            score_sum += p_good_value
            symbol_counts[current_symbol] = symbol_counts.get(current_symbol, 0) + 1
    frame["signal_flow_recent_signals_24h"] = recent_counts
    frame["signal_flow_recent_same_symbol_signals_24h"] = same_symbol_counts
    frame["signal_flow_recent_mean_detector_p_good_24h"] = recent_mean_scores
    return frame.drop(columns=["_stream_group_key"], errors="ignore")


def _stream_group_key(row: pd.Series) -> str:
    score_source = str(row["score_source"])
    fold_id = row["fold_id"]
    if pd.isna(fold_id):
        return score_source
    return f"{score_source}|{str(fold_id)}"


def _merge_required_rows(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list[str],
    source_name: str,
) -> pd.DataFrame:
    merged = left_df.merge(
        right_df,
        on=on,
        how="left",
        validate="many_to_one",
        indicator="_join_status",
    )
    missing_total = int((merged["_join_status"] == "left_only").sum())
    if missing_total > 0:
        raise ValueError(f"missing rows after join with {source_name}: missing_rows={missing_total}")
    return merged.drop(columns=["_join_status"], errors="ignore")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
