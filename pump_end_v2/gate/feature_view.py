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

GATE_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "detector_p_good",
    "detector_peak_p_good_before_fire",
    "detector_p_good_drop_from_peak",
    "detector_episode_age_bars",
    "detector_distance_from_episode_high_pct",
    "detector_policy_p_good_minus_fire_floor",
    "detector_policy_peak_minus_arm_min",
    "detector_policy_drop_minus_turn_down",
    "detector_policy_drop_to_turn_down_ratio",
    "detector_policy_p_good_band_position",
    "episode_runup_from_open_pct",
    "episode_extension_from_open_pct",
    "bars_since_episode_high",
    "drawdown_from_episode_high_so_far",
    "high_retest_count",
    "high_persistence_4",
    "episode_pump_context_streak",
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
    "strategy_recent_resolved_pnl_sum_24h",
    "strategy_recent_sl_rate_24h",
    "strategy_recent_tp_rate_24h",
    "strategy_prev_closed_losing_streak",
    "strategy_open_trades_now",
)

GATE_CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "cat_detector_band_position",
    "cat_episode_age_band",
    "cat_reference_regime",
    "cat_breadth_advancers_band",
    "cat_signal_crowding_24h_band",
    "cat_losing_streak_band",
)

GATE_FEATURE_COLUMNS: tuple[str, ...] = (
    *GATE_NUMERIC_FEATURE_COLUMNS,
    *GATE_CATEGORICAL_FEATURE_COLUMNS,
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
    "policy_arm_score_min",
    "policy_fire_score_floor",
    "policy_turn_down_delta",
    "episode_age_bars",
    "distance_from_episode_high_pct",
    "episode_runup_from_open_pct",
    "episode_extension_from_open_pct",
    "bars_since_episode_high",
    "drawdown_from_episode_high_so_far",
    "high_retest_count",
    "high_persistence_4",
    "episode_pump_context_streak",
)


def build_gate_feature_view(
    candidate_signals_df: pd.DataFrame,
    history_candidate_signals_df: pd.DataFrame | None,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        candidate_signals_df, _CANDIDATE_PROD_REQUIRED_COLUMNS, "candidate_signals_df"
    )
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
    frame = _prepare_candidate_frame(candidate_signals_df, history_context_only=False)
    if (
        history_candidate_signals_df is not None
        and not history_candidate_signals_df.empty
    ):
        history = _prepare_candidate_frame(
            history_candidate_signals_df, history_context_only=True
        )
        history = _normalize_history_stream_context(history, frame)
        frame = pd.concat([history, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["signal_id"], keep="last").reset_index(
            drop=True
        )
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
    breadth_part = breadth_state_df.rename(
        columns={"open_time": "context_bar_open_time"}
    )
    merged = _merge_required_rows(
        left_df=merged,
        right_df=breadth_part,
        on=["context_bar_open_time"],
        source_name="breadth_state_df",
    )
    merged["detector_p_good"] = pd.to_numeric(merged["p_good"], errors="coerce")
    merged["detector_peak_p_good_before_fire"] = pd.to_numeric(
        merged["peak_p_good_before_fire"], errors="coerce"
    )
    merged["detector_p_good_drop_from_peak"] = pd.to_numeric(
        merged["p_good_drop_from_peak"], errors="coerce"
    )
    merged["detector_episode_age_bars"] = pd.to_numeric(
        merged["episode_age_bars"], errors="coerce"
    )
    merged["detector_distance_from_episode_high_pct"] = pd.to_numeric(
        merged["distance_from_episode_high_pct"], errors="coerce"
    )
    policy_arm_score_min = pd.to_numeric(
        merged["policy_arm_score_min"], errors="coerce"
    )
    policy_fire_score_floor = pd.to_numeric(
        merged["policy_fire_score_floor"], errors="coerce"
    )
    policy_turn_down_delta = pd.to_numeric(
        merged["policy_turn_down_delta"], errors="coerce"
    )
    merged["detector_policy_p_good_minus_fire_floor"] = (
        merged["detector_p_good"] - policy_fire_score_floor
    )
    merged["detector_policy_peak_minus_arm_min"] = (
        merged["detector_peak_p_good_before_fire"] - policy_arm_score_min
    )
    merged["detector_policy_drop_minus_turn_down"] = (
        merged["detector_p_good_drop_from_peak"] - policy_turn_down_delta
    )
    safe_turn_down_delta = policy_turn_down_delta.where(
        policy_turn_down_delta.abs() > 1e-12
    )
    merged["detector_policy_drop_to_turn_down_ratio"] = (
        merged["detector_p_good_drop_from_peak"] / safe_turn_down_delta
    )
    policy_band_width = policy_arm_score_min - policy_fire_score_floor
    safe_policy_band_width = policy_band_width.where(policy_band_width.abs() > 1e-12)
    merged["detector_policy_p_good_band_position"] = (
        merged["detector_p_good"] - policy_fire_score_floor
    ) / safe_policy_band_width
    merged["episode_runup_from_open_pct"] = pd.to_numeric(
        merged["episode_runup_from_open_pct"], errors="coerce"
    )
    merged["episode_extension_from_open_pct"] = pd.to_numeric(
        merged["episode_extension_from_open_pct"], errors="coerce"
    )
    merged["bars_since_episode_high"] = pd.to_numeric(
        merged["bars_since_episode_high"], errors="coerce"
    )
    merged["drawdown_from_episode_high_so_far"] = pd.to_numeric(
        merged["drawdown_from_episode_high_so_far"], errors="coerce"
    )
    merged["high_retest_count"] = pd.to_numeric(
        merged["high_retest_count"], errors="coerce"
    )
    merged["high_persistence_4"] = pd.to_numeric(
        merged["high_persistence_4"], errors="coerce"
    )
    merged["episode_pump_context_streak"] = pd.to_numeric(
        merged["episode_pump_context_streak"], errors="coerce"
    )
    merged = _append_signal_flow_features(merged)
    merged = _append_strategy_state_features(merged)
    merged = _append_regime_categorical_features(merged)
    out = merged.loc[
        ~merged["_history_context_only"].astype(bool),
        [*GATE_IDENTITY_COLUMNS, *GATE_FEATURE_COLUMNS],
    ].copy()
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
    for _, idx in frame.groupby(
        "_stream_group_key", sort=False, dropna=False
    ).groups.items():
        history: deque[tuple[pd.Timestamp, str, float]] = deque()
        score_sum = 0.0
        symbol_counts: dict[str, int] = {}
        group_frame = frame.loc[list(idx), ["context_bar_open_time", "decision_time"]]
        for _, bucket_idx in group_frame.groupby(
            ["context_bar_open_time", "decision_time"], sort=False, dropna=False
        ).groups.items():
            bucket_indices = list(bucket_idx)
            now = pd.Timestamp(frame.iloc[bucket_indices[0]]["context_bar_open_time"])
            while history and history[0][0] < now - window:
                old_time, old_symbol, old_score = history.popleft()
                score_sum -= old_score
                current_count = symbol_counts.get(old_symbol, 0)
                if current_count <= 1:
                    symbol_counts.pop(old_symbol, None)
                else:
                    symbol_counts[old_symbol] = current_count - 1
            total_prev = len(history)
            additions: list[tuple[pd.Timestamp, str, float]] = []
            for row_idx in bucket_indices:
                row = frame.iloc[row_idx]
                current_symbol = str(row["symbol"])
                recent_counts[row_idx] = total_prev
                same_symbol_counts[row_idx] = symbol_counts.get(current_symbol, 0)
                recent_mean_scores[row_idx] = (
                    float(score_sum / total_prev) if total_prev > 0 else 0.0
                )
                p_good_value = float(
                    pd.to_numeric(row["detector_p_good"], errors="coerce")
                )
                if pd.isna(p_good_value):
                    p_good_value = 0.0
                additions.append((now, current_symbol, p_good_value))
            for add_time, add_symbol, add_score in additions:
                history.append((add_time, add_symbol, add_score))
                score_sum += add_score
                symbol_counts[add_symbol] = symbol_counts.get(add_symbol, 0) + 1
    frame["signal_flow_recent_signals_24h"] = recent_counts
    frame["signal_flow_recent_same_symbol_signals_24h"] = same_symbol_counts
    frame["signal_flow_recent_mean_detector_p_good_24h"] = recent_mean_scores
    return frame.drop(columns=["_stream_group_key"], errors="ignore")


def _append_strategy_state_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["_stream_group_key"] = frame.apply(_stream_group_key, axis=1)
    frame = frame.sort_values(
        ["_stream_group_key", "context_bar_open_time", "decision_time", "signal_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    required = {
        "counterfactual_trade_outcome",
        "counterfactual_trade_pnl_pct",
        "counterfactual_exit_time",
    }
    for column in required:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["counterfactual_trade_outcome"] = (
        frame["counterfactual_trade_outcome"].astype(str).str.lower()
    )
    frame["counterfactual_trade_pnl_pct"] = pd.to_numeric(
        frame["counterfactual_trade_pnl_pct"], errors="coerce"
    )
    frame["counterfactual_exit_time"] = pd.to_datetime(
        frame["counterfactual_exit_time"], utc=True, errors="coerce"
    )
    pnl_sum_24h = [0.0] * len(frame)
    sl_rate_24h = [0.0] * len(frame)
    tp_rate_24h = [0.0] * len(frame)
    losing_streak_prev = [0.0] * len(frame)
    open_trades_now = [0.0] * len(frame)
    window = pd.Timedelta(hours=24)
    for _, idx in frame.groupby(
        "_stream_group_key", sort=False, dropna=False
    ).groups.items():
        resolved_history: list[tuple[pd.Timestamp, float, str]] = []
        all_resolved: list[tuple[pd.Timestamp, str]] = []
        active_positions: list[tuple[pd.Timestamp | None, float, str]] = []
        group_frame = frame.loc[list(idx), ["context_bar_open_time", "decision_time"]]
        for _, bucket_idx in group_frame.groupby(
            ["context_bar_open_time", "decision_time"], sort=False, dropna=False
        ).groups.items():
            bucket_indices = list(bucket_idx)
            now = pd.Timestamp(frame.iloc[bucket_indices[0]]["context_bar_open_time"])
            resolved_history = [
                item for item in resolved_history if item[0] >= now - window
            ]
            still_active: list[tuple[pd.Timestamp | None, float, str]] = []
            for exit_time_item, pnl_item, outcome_item in active_positions:
                if exit_time_item is not None and exit_time_item < now:
                    resolved_history.append((exit_time_item, pnl_item, outcome_item))
                    all_resolved.append((exit_time_item, outcome_item))
                else:
                    still_active.append((exit_time_item, pnl_item, outcome_item))
            active_positions = still_active
            recent_total = len(resolved_history)
            if recent_total > 0:
                recent_pnl_sum = float(sum(item[1] for item in resolved_history))
                tp_count = int(sum(1 for item in resolved_history if item[2] == "tp"))
                sl_count = int(sum(1 for item in resolved_history if item[2] == "sl"))
                recent_tp_rate = float(tp_count / recent_total)
                recent_sl_rate = float(sl_count / recent_total)
            else:
                recent_pnl_sum = 0.0
                recent_tp_rate = 0.0
                recent_sl_rate = 0.0
            streak = 0
            sorted_resolved = sorted(all_resolved, key=lambda item: item[0])
            for _, outcome_flag in reversed(sorted_resolved):
                if outcome_flag == "sl":
                    streak += 1
                else:
                    break
            for row_idx in bucket_indices:
                pnl_sum_24h[row_idx] = recent_pnl_sum
                tp_rate_24h[row_idx] = recent_tp_rate
                sl_rate_24h[row_idx] = recent_sl_rate
                losing_streak_prev[row_idx] = float(streak)
                open_trades_now[row_idx] = float(len(active_positions))
            for row_idx in bucket_indices:
                row = frame.iloc[row_idx]
                exit_time = (
                    pd.Timestamp(row["counterfactual_exit_time"])
                    if pd.notna(row["counterfactual_exit_time"])
                    else None
                )
                outcome = str(row["counterfactual_trade_outcome"])
                pnl_value = (
                    float(row["counterfactual_trade_pnl_pct"])
                    if pd.notna(row["counterfactual_trade_pnl_pct"])
                    else 0.0
                )
                is_resolved = (
                    outcome in {"tp", "sl", "timeout", "ambiguous"}
                    and exit_time is not None
                )
                if is_resolved and exit_time < now:
                    resolved_history.append((exit_time, pnl_value, outcome))
                    all_resolved.append((exit_time, outcome))
                else:
                    active_positions.append((exit_time, pnl_value, outcome))
    frame["strategy_recent_resolved_pnl_sum_24h"] = pnl_sum_24h
    frame["strategy_recent_sl_rate_24h"] = sl_rate_24h
    frame["strategy_recent_tp_rate_24h"] = tp_rate_24h
    frame["strategy_prev_closed_losing_streak"] = losing_streak_prev
    frame["strategy_open_trades_now"] = open_trades_now
    return frame.drop(columns=["_stream_group_key"], errors="ignore")


def _append_regime_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    detector_band = pd.to_numeric(
        frame["detector_policy_p_good_band_position"], errors="coerce"
    )
    frame["cat_detector_band_position"] = detector_band.apply(
        _bucket_detector_band_position
    )
    episode_age = pd.to_numeric(frame["detector_episode_age_bars"], errors="coerce")
    frame["cat_episode_age_band"] = episode_age.apply(_bucket_episode_age_band)
    btc_ret_4 = pd.to_numeric(frame["btc_close_ret_4"], errors="coerce")
    eth_ret_4 = pd.to_numeric(frame["eth_close_ret_4"], errors="coerce")
    frame["cat_reference_regime"] = [
        _bucket_reference_regime(btc, eth) for btc, eth in zip(btc_ret_4, eth_ret_4)
    ]
    breadth_advancers = pd.to_numeric(
        frame["breadth_advancers_share"], errors="coerce"
    )
    frame["cat_breadth_advancers_band"] = breadth_advancers.apply(
        _bucket_breadth_advancers_band
    )
    signal_flow_24h = pd.to_numeric(
        frame["signal_flow_recent_signals_24h"], errors="coerce"
    )
    frame["cat_signal_crowding_24h_band"] = signal_flow_24h.apply(
        _bucket_signal_crowding_24h_band
    )
    losing_streak = pd.to_numeric(
        frame["strategy_prev_closed_losing_streak"], errors="coerce"
    )
    frame["cat_losing_streak_band"] = losing_streak.apply(_bucket_losing_streak_band)
    for column in GATE_CATEGORICAL_FEATURE_COLUMNS:
        frame[column] = frame[column].where(frame[column].notna(), "unknown")
        frame[column] = frame[column].astype(str).replace("", "unknown")
    return frame


def _bucket_detector_band_position(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    x = float(value)
    if x < 0.0:
        return "below_floor"
    if x < 0.33:
        return "low_band"
    if x < 0.66:
        return "mid_band"
    if x < 1.0:
        return "high_band"
    return "above_arm"


def _bucket_episode_age_band(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    x = int(value)
    if x <= 1:
        return "age_0_1"
    if x <= 3:
        return "age_2_3"
    if x <= 6:
        return "age_4_6"
    return "age_7_plus"


def _bucket_reference_regime(btc_ret_4: float, eth_ret_4: float) -> str:
    if pd.isna(btc_ret_4) or pd.isna(eth_ret_4):
        return "unknown"
    btc_up = float(btc_ret_4) >= 0.0
    eth_up = float(eth_ret_4) >= 0.0
    if not btc_up and not eth_up:
        return "both_down"
    if not btc_up and eth_up:
        return "btc_down_eth_up"
    if btc_up and not eth_up:
        return "btc_up_eth_down"
    return "both_up"


def _bucket_breadth_advancers_band(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    x = float(value)
    if x < 0.35:
        return "breadth_weak"
    if x <= 0.65:
        return "breadth_neutral"
    return "breadth_strong"


def _bucket_signal_crowding_24h_band(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    x = int(value)
    if x <= 0:
        return "flow_0"
    if x == 1:
        return "flow_1"
    if x <= 3:
        return "flow_2_3"
    return "flow_4_plus"


def _bucket_losing_streak_band(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    x = int(value)
    if x <= 0:
        return "streak_0"
    if x == 1:
        return "streak_1"
    if x == 2:
        return "streak_2"
    return "streak_3_plus"


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
        raise ValueError(
            f"missing rows after join with {source_name}: missing_rows={missing_total}"
        )
    return merged.drop(columns=["_join_status"], errors="ignore")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _prepare_candidate_frame(
    candidate_signals_df: pd.DataFrame, history_context_only: bool
) -> pd.DataFrame:
    frame = candidate_signals_df.copy()
    frame["context_bar_open_time"] = pd.to_datetime(
        frame["context_bar_open_time"], utc=True, errors="raise"
    )
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"], utc=True, errors="raise"
    )
    frame["entry_bar_open_time"] = pd.to_datetime(
        frame["entry_bar_open_time"], utc=True, errors="raise"
    )
    if "score_source" not in frame.columns:
        frame["score_source"] = "unknown"
    if "fold_id" not in frame.columns:
        frame["fold_id"] = pd.NA
    frame["_history_context_only"] = bool(history_context_only)
    return frame


def _normalize_history_stream_context(
    history_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    if history_df.empty or current_df.empty:
        return history_df
    current_score_source = current_df["score_source"].dropna().astype(str).mode()
    if not current_score_source.empty:
        history_df["score_source"] = str(current_score_source.iloc[0])
    history_df["fold_id"] = pd.NA
    return history_df
