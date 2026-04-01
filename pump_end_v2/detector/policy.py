from typing import Any

import pandas as pd

from pump_end_v2.config import DetectorPolicyConfig
from pump_end_v2.contracts import CandidateSignalRef
from pump_end_v2.logging import log_info

INPUT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "p_good",
    "score_source",
    "policy_context_only",
    "episode_age_bars",
    "distance_from_episode_high_pct",
)

_OPTIONAL_HINDSIGHT_COLUMNS: tuple[str, ...] = (
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "signal_quality_h32",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
    "bars_to_pullback",
    "bars_to_peak_after_row",
    "bars_to_resolution",
    "entry_quality_score",
    "ideal_entry_row_id",
    "ideal_entry_bar_open_time",
    "is_ideal_entry",
)

CANDIDATE_LEDGER_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "episode_id",
    "symbol",
    "fire_decision_row_id",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "episode_age_bars",
    "p_good",
    "peak_p_good_before_fire",
    "p_good_drop_from_peak",
    "distance_from_episode_high_pct",
    "episode_runup_from_open_pct",
    "episode_extension_from_open_pct",
    "bars_since_episode_high",
    "drawdown_from_episode_high_so_far",
    "high_retest_count",
    "high_persistence_4",
    "episode_pump_context_streak",
    "score_source",
    "fold_id",
    "policy_arm_score_min",
    "policy_fire_score_floor",
    "policy_turn_down_delta",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
    "signal_quality_h32",
    "future_prepullback_squeeze_pct",
    "future_pullback_pct",
    "future_net_edge_pct",
    "bars_to_pullback",
    "bars_to_peak_after_row",
    "bars_to_resolution",
    "entry_quality_score",
    "ideal_entry_row_id",
    "ideal_entry_bar_open_time",
    "is_ideal_entry",
    "bars_fire_to_ideal",
)

EPISODE_POLICY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "symbol",
    "score_source",
    "fold_id",
    "active_rows_total",
    "good_episode_flag",
    "armed",
    "fired",
    "reset_count",
    "fire_signal_id",
    "fire_target_good_short_now",
    "fire_target_reason",
    "fire_future_outcome_class",
    "fire_signal_quality_h32",
    "fire_future_net_edge_pct",
    "bars_fire_to_ideal",
)


def apply_episode_aware_detector_policy(
    scored_rows_df: pd.DataFrame,
    detector_policy_config: DetectorPolicyConfig,
    emit_summary_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime_cache = build_detector_policy_runtime_cache(scored_rows_df)
    return apply_episode_aware_detector_policy_cached(
        runtime_cache=runtime_cache,
        detector_policy_config=detector_policy_config,
        emit_summary_log=emit_summary_log,
    )


def build_detector_policy_runtime_cache(scored_rows_df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(scored_rows_df, INPUT_REQUIRED_COLUMNS)
    has_fold_id = "fold_id" in scored_rows_df.columns
    frame = scored_rows_df.copy()
    frame["context_bar_open_time"] = pd.to_datetime(
        frame["context_bar_open_time"], utc=True, errors="raise"
    )
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"], utc=True, errors="raise"
    )
    frame["entry_bar_open_time"] = pd.to_datetime(
        frame["entry_bar_open_time"], utc=True, errors="raise"
    )
    if "ideal_entry_bar_open_time" in frame.columns:
        frame["ideal_entry_bar_open_time"] = pd.to_datetime(
            frame["ideal_entry_bar_open_time"], utc=True, errors="coerce"
        )
    frame["policy_context_only"] = frame["policy_context_only"].astype(bool)
    frame["p_good"] = frame["p_good"].astype(float)
    if not has_fold_id:
        frame["fold_id"] = pd.NA
    sort_columns = ["score_source", "episode_id", "context_bar_open_time"]
    if has_fold_id:
        sort_columns = [
            "score_source",
            "fold_id",
            "episode_id",
            "context_bar_open_time",
        ]
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    group_columns = ["score_source", "episode_id"]
    if has_fold_id:
        group_columns = ["score_source", "fold_id", "episode_id"]
    prepared_groups: list[dict[str, Any]] = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        prepared = group.reset_index(drop=True)
        prepared_groups.append(
            {
                "rows_tuples": list(prepared.itertuples(index=False)),
                "symbol": str(prepared["symbol"].iloc[0]),
                "episode_id": str(prepared["episode_id"].iloc[0]),
                "score_source": str(prepared["score_source"].iloc[0]),
                "fold_id": prepared["fold_id"].iloc[0] if has_fold_id else pd.NA,
                "active_rows_total": int((~prepared["policy_context_only"]).sum()),
                "good_episode_flag": _compute_good_episode_flag(
                    prepared.loc[~prepared["policy_context_only"]]
                ),
            }
        )
    return {"prepared_groups": prepared_groups}


def apply_episode_aware_detector_policy_cached(
    runtime_cache: dict[str, Any],
    detector_policy_config: DetectorPolicyConfig,
    emit_summary_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for prepared_group in runtime_cache["prepared_groups"]:
        symbol = prepared_group["symbol"]
        episode_id = prepared_group["episode_id"]
        score_source = prepared_group["score_source"]
        fold_id = prepared_group["fold_id"]
        active_rows_total = prepared_group["active_rows_total"]
        if active_rows_total == 0:
            continue
        good_episode_flag = prepared_group["good_episode_flag"]
        armed_flag = False
        had_arm = False
        fired = False
        expired = False
        reset_count = 0
        peak_p_good: float | None = None
        fire_signal_row: dict[str, Any] | None = None
        for row in prepared_group["rows_tuples"]:
            if fired or expired:
                continue
            current_p_good = float(row.p_good)
            if current_p_good >= detector_policy_config.arm_score_min:
                if not armed_flag:
                    armed_flag = True
                    had_arm = True
                    peak_p_good = current_p_good
                elif peak_p_good is not None and current_p_good > peak_p_good:
                    peak_p_good = current_p_good
            if not armed_flag or peak_p_good is None:
                continue
            drop_from_peak = float(peak_p_good - current_p_good)
            if (
                drop_from_peak >= detector_policy_config.turn_down_delta
                and current_p_good >= detector_policy_config.fire_score_floor
            ):
                if bool(row.policy_context_only):
                    expired = True
                    continue
                signal_id = f"{episode_id}|{pd.Timestamp(row.context_bar_open_time).strftime('%Y%m%d_%H%M%S')}"
                _ = CandidateSignalRef(
                    signal_id=signal_id,
                    episode_id=episode_id,
                    symbol=symbol,
                    fire_decision_time=pd.Timestamp(row.decision_time).to_pydatetime(),
                    entry_bar_open_time=pd.Timestamp(
                        row.entry_bar_open_time
                    ).to_pydatetime(),
                )
                ideal_entry_value = _row_value(row, "ideal_entry_bar_open_time")
                ideal_entry_time = (
                    pd.Timestamp(ideal_entry_value)
                    if pd.notna(ideal_entry_value)
                    else None
                )
                bars_fire_to_ideal = _compute_bars_fire_to_ideal(
                    pd.Timestamp(row.entry_bar_open_time),
                    ideal_entry_time,
                )
                fire_signal_row = {
                    "signal_id": signal_id,
                    "episode_id": episode_id,
                    "symbol": symbol,
                    "fire_decision_row_id": str(row.decision_row_id),
                    "context_bar_open_time": pd.Timestamp(row.context_bar_open_time),
                    "decision_time": pd.Timestamp(row.decision_time),
                    "entry_bar_open_time": pd.Timestamp(row.entry_bar_open_time),
                    "episode_age_bars": row.episode_age_bars,
                    "p_good": current_p_good,
                    "peak_p_good_before_fire": peak_p_good,
                    "p_good_drop_from_peak": drop_from_peak,
                    "distance_from_episode_high_pct": row.distance_from_episode_high_pct,
                    "episode_runup_from_open_pct": _row_value(
                        row, "episode_runup_from_open_pct"
                    ),
                    "episode_extension_from_open_pct": _row_value(
                        row, "episode_extension_from_open_pct"
                    ),
                    "bars_since_episode_high": _row_value(
                        row, "bars_since_episode_high"
                    ),
                    "drawdown_from_episode_high_so_far": _row_value(
                        row, "drawdown_from_episode_high_so_far"
                    ),
                    "high_retest_count": _row_value(row, "high_retest_count"),
                    "high_persistence_4": _row_value(row, "high_persistence_4"),
                    "episode_pump_context_streak": _row_value(
                        row, "episode_pump_context_streak"
                    ),
                    "score_source": score_source,
                    "fold_id": fold_id,
                    "policy_arm_score_min": detector_policy_config.arm_score_min,
                    "policy_fire_score_floor": detector_policy_config.fire_score_floor,
                    "policy_turn_down_delta": detector_policy_config.turn_down_delta,
                    "bars_fire_to_ideal": bars_fire_to_ideal,
                }
                for column in _OPTIONAL_HINDSIGHT_COLUMNS:
                    fire_signal_row[column] = _row_value(row, column)
                candidate_rows.append(fire_signal_row)
                fired = True
                continue
            if current_p_good < detector_policy_config.fire_score_floor:
                armed_flag = False
                peak_p_good = None
                reset_count += 1
        summary_rows.append(
            {
                "episode_id": episode_id,
                "symbol": symbol,
                "score_source": score_source,
                "fold_id": fold_id,
                "active_rows_total": active_rows_total,
                "good_episode_flag": good_episode_flag,
                "armed": had_arm,
                "fired": fired,
                "reset_count": reset_count,
                "fire_signal_id": (
                    fire_signal_row["signal_id"] if fire_signal_row else pd.NA
                ),
                "fire_target_good_short_now": (
                    fire_signal_row["target_good_short_now"]
                    if fire_signal_row
                    else pd.NA
                ),
                "fire_target_reason": (
                    fire_signal_row["target_reason"] if fire_signal_row else pd.NA
                ),
                "fire_future_outcome_class": (
                    fire_signal_row["future_outcome_class"]
                    if fire_signal_row
                    else pd.NA
                ),
                "fire_signal_quality_h32": (
                    fire_signal_row["signal_quality_h32"] if fire_signal_row else pd.NA
                ),
                "fire_future_net_edge_pct": (
                    fire_signal_row["future_net_edge_pct"] if fire_signal_row else pd.NA
                ),
                "bars_fire_to_ideal": (
                    fire_signal_row["bars_fire_to_ideal"] if fire_signal_row else pd.NA
                ),
            }
        )
    candidate_signals_df = pd.DataFrame(candidate_rows)
    if candidate_signals_df.empty:
        candidate_signals_df = pd.DataFrame(columns=list(CANDIDATE_LEDGER_COLUMNS))
    else:
        candidate_signals_df = candidate_signals_df.loc[
            :, list(CANDIDATE_LEDGER_COLUMNS)
        ].copy()
    episode_policy_summary_df = pd.DataFrame(summary_rows)
    if episode_policy_summary_df.empty:
        episode_policy_summary_df = pd.DataFrame(
            columns=list(EPISODE_POLICY_SUMMARY_COLUMNS)
        )
    else:
        episode_policy_summary_df = episode_policy_summary_df.loc[
            :, list(EPISODE_POLICY_SUMMARY_COLUMNS)
        ].copy()
    if emit_summary_log:
        log_info(
            "POLICY",
            f"policy apply done episodes_total={len(episode_policy_summary_df)} signals_total={len(candidate_signals_df)}",
        )
    return candidate_signals_df, episode_policy_summary_df


def build_detector_policy_metrics(
    candidate_signals_df: pd.DataFrame,
    episode_policy_summary_df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> dict[str, float]:
    _require_columns(episode_policy_summary_df, EPISODE_POLICY_SUMMARY_COLUMNS)
    episodes_total = int(len(episode_policy_summary_df))
    good_mask = episode_policy_summary_df["good_episode_flag"].astype(bool)
    fired_mask = episode_policy_summary_df["fired"].astype(bool)
    armed_mask = episode_policy_summary_df["armed"].astype(bool)
    episodes_with_good_zone = int(good_mask.sum())
    episodes_armed = int(armed_mask.sum())
    episodes_fired = int(fired_mask.sum())
    good_episode_capture_rate = _safe_ratio(
        int((good_mask & fired_mask).sum()), episodes_with_good_zone
    )
    bad_mask = ~good_mask
    bad_episode_fire_rate = _safe_ratio(
        int((bad_mask & fired_mask).sum()), int(bad_mask.sum())
    )
    fired_good_rate = (
        float(
            pd.to_numeric(
                candidate_signals_df.get("target_good_short_now"), errors="coerce"
            ).mean()
        )
        if len(candidate_signals_df) > 0
        else 0.0
    )
    eval_window_days = _resolve_eval_window_days(
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
    )
    fires_per_30d = _compute_fires_per_30d(len(candidate_signals_df), eval_window_days)
    median_bars_fire_to_ideal = (
        float(
            pd.to_numeric(
                candidate_signals_df.get("bars_fire_to_ideal"), errors="coerce"
            ).median()
        )
        if len(candidate_signals_df) > 0
        else float("nan")
    )
    median_future_net_edge_pct_at_fire = (
        float(
            pd.to_numeric(
                candidate_signals_df.get("future_net_edge_pct"), errors="coerce"
            ).median()
        )
        if len(candidate_signals_df) > 0
        else float("nan")
    )
    reset_without_fire_share = _safe_ratio(
        int(
            (
                (episode_policy_summary_df["reset_count"].fillna(0).astype(int) > 0)
                & ~fired_mask
            ).sum()
        ),
        episodes_total,
    )
    arm_to_fire_conversion = _safe_ratio(episodes_fired, episodes_armed)
    density_sanity_penalty = _compute_detector_density_sanity_penalty(fires_per_30d)
    selection_score = (
        good_episode_capture_rate
        + fired_good_rate
        - bad_episode_fire_rate
        - 0.15 * density_sanity_penalty
    )
    return {
        "episodes_total": float(episodes_total),
        "episodes_with_good_zone": float(episodes_with_good_zone),
        "episodes_armed": float(episodes_armed),
        "episodes_fired": float(episodes_fired),
        "good_episode_capture_rate": float(good_episode_capture_rate),
        "bad_episode_fire_rate": float(bad_episode_fire_rate),
        "fired_good_rate": float(fired_good_rate),
        "fires_per_30d": float(fires_per_30d),
        "median_bars_fire_to_ideal": float(median_bars_fire_to_ideal),
        "median_future_net_edge_pct_at_fire": float(median_future_net_edge_pct_at_fire),
        "reset_without_fire_share": float(reset_without_fire_share),
        "arm_to_fire_conversion": float(arm_to_fire_conversion),
        "density_sanity_penalty": float(density_sanity_penalty),
        "selection_score": float(selection_score),
    }


def compute_eval_window_days_from_policy_rows(policy_rows_df: pd.DataFrame) -> float:
    if "context_bar_open_time" not in policy_rows_df.columns:
        return 1.0
    frame = policy_rows_df.copy()
    if "policy_context_only" in frame.columns:
        mask = ~frame["policy_context_only"].astype(bool)
        frame = frame[mask].copy()
    if frame.empty:
        return 1.0
    timestamps = pd.to_datetime(
        frame["context_bar_open_time"], utc=True, errors="coerce"
    ).dropna()
    unique_active_timestamps = int(pd.Index(timestamps).nunique())
    if unique_active_timestamps <= 0:
        return 1.0
    bars_per_day = float(pd.Timedelta(days=1) / pd.Timedelta(minutes=15))
    return float(unique_active_timestamps / bars_per_day)


def _compute_bars_fire_to_ideal(
    entry_bar_open_time: pd.Timestamp,
    ideal_entry_bar_open_time: pd.Timestamp | None,
) -> float:
    if ideal_entry_bar_open_time is None or pd.isna(ideal_entry_bar_open_time):
        return float("nan")
    return float(
        (entry_bar_open_time - ideal_entry_bar_open_time) / pd.Timedelta(minutes=15)
    )


def _compute_fires_per_30d(fires_count: int, eval_window_days: float) -> float:
    if fires_count <= 0:
        return 0.0
    safe_window_days = max(float(eval_window_days), 1e-9)
    return float(float(fires_count) * 30.0 / safe_window_days)


def _resolve_eval_window_days(
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
    window_days: float | None,
) -> float:
    if window_days is not None:
        resolved = float(window_days)
        if resolved <= 0.0:
            raise ValueError("window_days must be positive")
        return resolved
    if window_start is None or window_end is None:
        return 1.0
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    if end <= start:
        raise ValueError("window_end must be greater than window_start")
    return float((end - start) / pd.Timedelta(days=1))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _compute_detector_density_sanity_penalty(fires_per_30d: float) -> float:
    value = float(fires_per_30d)
    if 15.0 <= value <= 180.0:
        return 0.0
    if value < 15.0:
        return float((15.0 - value) / 15.0)
    return float((value - 180.0) / 180.0)


def _compute_good_episode_flag(active_rows: pd.DataFrame) -> bool:
    if "signal_quality_h32" in active_rows.columns:
        quality = active_rows["signal_quality_h32"].astype(str)
        return bool((quality == "clean_retrace_h32").any())
    if "target_good_short_now" in active_rows.columns:
        return bool(
            (
                pd.to_numeric(active_rows["target_good_short_now"], errors="coerce")
                == 1
            ).any()
        )
    return False


def _row_value(row: Any, column: str) -> Any:
    return getattr(row, column) if hasattr(row, column) else pd.NA


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing required columns: {missing}")
