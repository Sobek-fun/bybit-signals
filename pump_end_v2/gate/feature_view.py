from __future__ import annotations

from collections import deque
import heapq

import pandas as pd

from pump_end_v2.contracts import ExecutionContract
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
    "recent_resolved_tp_rate_24h",
    "recent_resolved_sl_rate_24h",
    "recent_pnl_sum_24h",
    "recent_pnl_sum_last_n",
    "recent_losing_streak",
    "active_symbol_sessions_now",
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
    execution_contract: ExecutionContract | None = None,
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
    merged = _append_strategy_state_features(merged, execution_contract)
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


def _append_strategy_state_features(
    df: pd.DataFrame,
    execution_contract: ExecutionContract | None,
) -> pd.DataFrame:
    frame = df.copy()
    frame["_stream_group_key"] = frame.apply(_stream_group_key, axis=1)
    frame = frame.sort_values(
        ["_stream_group_key", "context_bar_open_time", "decision_time", "signal_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    tp_rate_24h = [0.0] * len(frame)
    sl_rate_24h = [0.0] * len(frame)
    pnl_sum_24h = [0.0] * len(frame)
    pnl_sum_last_n = [0.0] * len(frame)
    losing_streak = [0.0] * len(frame)
    active_symbol_sessions = [0.0] * len(frame)
    window = pd.Timedelta(hours=24)
    last_n = 10
    tp_pct = float(execution_contract.tp_pct) * 100.0 if execution_contract is not None else 4.5
    sl_pct = float(execution_contract.sl_pct) * 100.0 if execution_contract is not None else 3.0
    max_hold_bars = int(execution_contract.max_hold_bars) if execution_contract is not None else 96
    has_hindsight_columns = all(column in frame.columns for column in ("signal_quality_h32", "bars_to_resolution"))
    for _, idx in frame.groupby("_stream_group_key", sort=False, dropna=False).groups.items():
        event_seq = 0
        start_heap: list[tuple[int, int, str, int, int, float]] = []
        end_heap: list[tuple[int, int, str]] = []
        resolve_heap: list[tuple[int, int, float, int, int]] = []
        history_24h: deque[tuple[pd.Timestamp, float, int, int]] = deque()
        pnl_history_last_n: deque[float] = deque()
        running_pnl_24h = 0.0
        running_tp_24h = 0
        running_sl_24h = 0
        symbol_counts: dict[str, int] = {}
        streak = 0
        for row_idx in idx:
            row = frame.iloc[row_idx]
            now = pd.Timestamp(row["context_bar_open_time"])
            now_ns = int(now.value)
            while start_heap and start_heap[0][0] <= now_ns:
                _, _, start_symbol, close_ns, close_seq, close_pnl = heapq.heappop(start_heap)
                symbol_counts[start_symbol] = symbol_counts.get(start_symbol, 0) + 1
                heapq.heappush(end_heap, (close_ns, close_seq, start_symbol))
                if close_pnl != 0.0:
                    tp_event = 1 if close_pnl > 0 else 0
                    sl_event = 1 if close_pnl < 0 else 0
                    heapq.heappush(resolve_heap, (close_ns, close_seq, close_pnl, tp_event, sl_event))
            while end_heap and end_heap[0][0] <= now_ns:
                _, _, end_symbol = heapq.heappop(end_heap)
                count = symbol_counts.get(end_symbol, 0)
                if count <= 1:
                    symbol_counts.pop(end_symbol, None)
                else:
                    symbol_counts[end_symbol] = count - 1
            while resolve_heap and resolve_heap[0][0] <= now_ns:
                close_ns, _, close_pnl, tp_event, sl_event = heapq.heappop(resolve_heap)
                close_time = pd.Timestamp(close_ns, tz="UTC")
                history_24h.append((close_time, close_pnl, tp_event, sl_event))
                running_pnl_24h += close_pnl
                running_tp_24h += tp_event
                running_sl_24h += sl_event
                pnl_history_last_n.append(close_pnl)
                while len(pnl_history_last_n) > last_n:
                    pnl_history_last_n.popleft()
                if close_pnl < 0:
                    streak += 1
                else:
                    streak = 0
            while history_24h and history_24h[0][0] < now - window:
                _, old_pnl, old_tp, old_sl = history_24h.popleft()
                running_pnl_24h -= old_pnl
                running_tp_24h -= old_tp
                running_sl_24h -= old_sl
            resolved_total = running_tp_24h + running_sl_24h
            tp_rate_24h[row_idx] = float(running_tp_24h / resolved_total) if resolved_total > 0 else 0.0
            sl_rate_24h[row_idx] = float(running_sl_24h / resolved_total) if resolved_total > 0 else 0.0
            pnl_sum_24h[row_idx] = float(running_pnl_24h)
            pnl_sum_last_n[row_idx] = float(sum(pnl_history_last_n))
            losing_streak[row_idx] = float(streak)
            active_symbol_sessions[row_idx] = float(len(symbol_counts))
            if not has_hindsight_columns:
                continue
            quality = str(row["signal_quality_h32"])
            bars_to_resolution = pd.to_numeric(row["bars_to_resolution"], errors="coerce")
            quality_valid = quality not in {"", "nan", "<NA>"}
            bars_valid = pd.notna(bars_to_resolution) and float(bars_to_resolution) >= 1.0
            if not (quality_valid and bars_valid):
                continue
            is_bad = quality in {
                "dirty_retrace_h32",
                "clean_no_pullback_h32",
                "dirty_no_pullback_h32",
                "pullback_before_squeeze_h32",
            }
            close_pnl = tp_pct if quality == "clean_retrace_h32" else (-sl_pct if is_bad else 0.0)
            entry_time = pd.Timestamp(row["entry_bar_open_time"])
            hold_bars = min(max_hold_bars, max(1, int(float(bars_to_resolution))))
            close_time = entry_time + pd.Timedelta(minutes=15 * hold_bars)
            event_seq += 1
            heapq.heappush(
                start_heap,
                (
                    int(entry_time.value),
                    event_seq,
                    str(row["symbol"]),
                    int(close_time.value),
                    event_seq,
                    float(close_pnl),
                ),
            )
    frame["recent_resolved_tp_rate_24h"] = tp_rate_24h
    frame["recent_resolved_sl_rate_24h"] = sl_rate_24h
    frame["recent_pnl_sum_24h"] = pnl_sum_24h
    frame["recent_pnl_sum_last_n"] = pnl_sum_last_n
    frame["recent_losing_streak"] = losing_streak
    frame["active_symbol_sessions_now"] = active_symbol_sessions
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
