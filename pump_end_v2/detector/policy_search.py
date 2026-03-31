from itertools import product
import time

import pandas as pd
from catboost import CatBoostClassifier

from pump_end_v2.config import (
    DetectorCVConfig,
    DetectorModelConfig,
    DetectorPolicyConfig,
    DetectorPolicySearchConfig,
    EventOpenerConfig,
    ResolverConfig,
    SplitBounds,
)
from pump_end_v2.detector.model import (
    build_detector_model,
    fit_detector_model,
    predict_detector_scores,
)
from pump_end_v2.detector.policy import (
    apply_episode_aware_detector_policy,
    build_detector_policy_metrics,
    compute_eval_window_days_from_policy_rows,
)
from pump_end_v2.detector.splits import (
    filter_fold_rows,
    generate_detector_walkforward_folds,
)
from pump_end_v2.features.manifest import DETECTOR_FEATURE_COLUMNS
from pump_end_v2.logging import log_info

POLICY_ROW_COLUMNS: tuple[str, ...] = (
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "p_good",
    "p_too_early",
    "p_too_late",
    "p_continuation_flat",
    "predicted_reason_group",
    "predicted_reason_group_id",
    "score_source",
    "fold_id",
    "policy_context_only",
    "episode_age_bars",
    "distance_from_episode_high_pct",
    "episode_runup_from_open_pct",
    "episode_extension_from_open_pct",
    "bars_since_episode_high",
    "drawdown_from_episode_high_so_far",
    "high_retest_count",
    "high_persistence_4",
    "episode_pump_context_streak",
    "target_good_short_now",
    "target_reason",
    "target_reason_group",
    "target_reason_group_id",
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

SWEEP_COLUMNS: tuple[str, ...] = (
    "arm_score_min",
    "fire_score_floor",
    "turn_down_delta",
    "max_too_early_prob",
    "max_too_late_prob",
    "episodes_total",
    "episodes_with_good_zone",
    "episodes_fired",
    "good_episode_capture_rate",
    "bad_episode_fire_rate",
    "fired_good_rate",
    "fires_per_30d",
    "median_bars_fire_to_ideal",
    "median_future_net_edge_pct_at_fire",
    "reset_without_fire_share",
    "arm_to_fire_conversion",
    "density_sanity_penalty",
    "selection_score",
)

POLICY_BASE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset_split",
    "trainable_row",
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "episode_age_bars",
    "distance_from_episode_high_pct",
    "target_good_short_now",
    "target_reason_group",
    "target_reason_group_id",
    *DETECTOR_FEATURE_COLUMNS,
)


def build_detector_val_policy_rows(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy val scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    purge_gap_timedelta = pd.Timedelta(minutes=15 * resolver_config.horizon_bars)
    effective_train_end = pd.Timestamp(split_bounds.train_end) - purge_gap_timedelta
    train_fit = frame[
        (frame["dataset_split"] == "train")
        & frame["trainable_row"].astype(bool)
        & (frame["context_bar_open_time"] <= effective_train_end)
        ].copy()
    if train_fit.empty:
        raise ValueError(
            "no trainable train rows available for detector val policy scoring"
        )
    model = build_detector_model(detector_model_config)
    fit_detector_model(
        model, train_fit, DETECTOR_FEATURE_COLUMNS, "target_reason_group_id"
    )
    val_rows = frame[frame["dataset_split"] == "val"].copy()
    if val_rows.empty:
        scored = pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
        log_info(
            "POLICY",
            "policy val scoring rows build done rows_total=0 context_rows=0 active_rows=0",
        )
        return model, scored
    active_window_end = pd.Timestamp(split_bounds.val_end)
    val_active = val_rows[
        _build_active_eligibility_mask(
            val_rows,
            resolver_config=resolver_config,
            active_window_end=active_window_end,
        )
    ].copy()
    if val_active.empty:
        scored = pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
        log_info(
            "POLICY",
            "policy val scoring rows build done rows_total=0 context_rows=0 active_rows=0",
        )
        return model, scored
    active_start = val_active["context_bar_open_time"].min()
    warmup_timedelta = pd.Timedelta(minutes=15 * event_opener_config.max_episode_bars)
    warmup_start = active_start - warmup_timedelta
    warmup_rows = frame[
        (frame["context_bar_open_time"] >= warmup_start)
        & (frame["context_bar_open_time"] < active_start)
        ].copy()
    warmup_rows["policy_context_only"] = True
    val_active["policy_context_only"] = False
    score_window = pd.concat([warmup_rows, val_active], ignore_index=True)
    score_window = score_window.drop_duplicates(
        subset=["decision_row_id"], keep="last"
    ).reset_index(drop=True)
    scored = _score_policy_window(
        model=model,
        rows_to_score=score_window,
        score_source="val_forward",
        fold_id=pd.NA,
    )
    context_rows = int(scored["policy_context_only"].sum()) if len(scored) > 0 else 0
    active_rows = int((~scored["policy_context_only"]).sum()) if len(scored) > 0 else 0
    log_info(
        "POLICY",
        (
            "policy val scoring rows build done "
            f"rows_total={len(scored)} context_rows={context_rows} active_rows={active_rows}"
        ),
    )
    return model, scored


def build_detector_test_policy_rows(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy test scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    purge_gap_timedelta = pd.Timedelta(minutes=15 * resolver_config.horizon_bars)
    effective_train_end = pd.Timestamp(split_bounds.train_end) - purge_gap_timedelta
    train_fit = frame[
        (frame["dataset_split"] == "train")
        & frame["trainable_row"].astype(bool)
        & (frame["context_bar_open_time"] <= effective_train_end)
        ].copy()
    if train_fit.empty:
        raise ValueError(
            "no trainable train rows available for detector test policy scoring"
        )
    model = build_detector_model(detector_model_config)
    fit_detector_model(
        model, train_fit, DETECTOR_FEATURE_COLUMNS, "target_reason_group_id"
    )
    test_rows = frame[frame["dataset_split"] == "test"].copy()
    if test_rows.empty:
        scored = pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
        log_info(
            "POLICY",
            "detector test policy rows build done rows_total=0 context_rows=0 active_rows=0",
        )
        return model, scored
    active_window_end = pd.Timestamp(split_bounds.test_end)
    test_active = test_rows[
        _build_active_eligibility_mask(
            test_rows,
            resolver_config=resolver_config,
            active_window_end=active_window_end,
        )
    ].copy()
    if test_active.empty:
        scored = pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
        log_info(
            "POLICY",
            "detector test policy rows build done rows_total=0 context_rows=0 active_rows=0",
        )
        return model, scored
    active_start = test_active["context_bar_open_time"].min()
    warmup_timedelta = pd.Timedelta(minutes=15 * event_opener_config.max_episode_bars)
    warmup_start = active_start - warmup_timedelta
    warmup_rows = frame[
        (frame["context_bar_open_time"] >= warmup_start)
        & (frame["context_bar_open_time"] < active_start)
        ].copy()
    warmup_rows["policy_context_only"] = True
    test_active["policy_context_only"] = False
    score_window = pd.concat([warmup_rows, test_active], ignore_index=True)
    score_window = score_window.drop_duplicates(
        subset=["decision_row_id"], keep="last"
    ).reset_index(drop=True)
    scored = _score_policy_window(
        model=model,
        rows_to_score=score_window,
        score_source="test_forward",
        fold_id=pd.NA,
    )
    context_rows = int(scored["policy_context_only"].sum()) if len(scored) > 0 else 0
    active_rows = int((~scored["policy_context_only"]).sum()) if len(scored) > 0 else 0
    log_info(
        "POLICY",
        (
            "detector test policy rows build done "
            f"rows_total={len(scored)} context_rows={context_rows} active_rows={active_rows}"
        ),
    )
    return model, scored


def build_detector_train_oof_policy_rows(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_cv_config: DetectorCVConfig,
        detector_model_config: DetectorModelConfig,
) -> pd.DataFrame:
    started = time.perf_counter()
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy OOF scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    folds = generate_detector_walkforward_folds(
        frame, split_bounds, resolver_config, detector_cv_config
    )
    warmup_timedelta = pd.Timedelta(minutes=15 * event_opener_config.max_episode_bars)
    chunks: list[pd.DataFrame] = []
    for idx, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        log_info(
            "POLICY",
            (
                f"oof fold start fold={idx}/{len(folds)} fold_id={fold.fold_id} "
                f"train_rows={fold.train_row_count} val_rows={fold.val_row_count} "
                f"train_start={pd.Timestamp(fold.train_start)} train_end={pd.Timestamp(fold.train_end)} "
                f"val_start={pd.Timestamp(fold.val_start)} val_end={pd.Timestamp(fold.val_end)}"
            ),
        )
        fold_train_raw, _ = filter_fold_rows(frame, fold)
        fold_train_fit = fold_train_raw[
            fold_train_raw["trainable_row"].astype(bool)
        ].copy()
        if fold_train_fit.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{len(folds)} reason=no_trainable_rows",
            )
            continue
        model = build_detector_model(detector_model_config)
        fit_started = time.perf_counter()
        fit_detector_model(
            model, fold_train_fit, DETECTOR_FEATURE_COLUMNS, "target_reason_group_id"
        )
        log_info(
            "POLICY",
            (
                f"oof fold model_fit done fold={idx}/{len(folds)} "
                f"rows_fit={len(fold_train_fit)} elapsed_sec_fit={time.perf_counter() - fit_started:.3f}"
            ),
        )
        val_start = pd.Timestamp(fold.val_start)
        val_end = pd.Timestamp(fold.val_end)
        warmup_start = val_start - warmup_timedelta
        fold_active_rows = frame[
            (frame["dataset_split"] == "train")
            & (frame["context_bar_open_time"] >= val_start)
            & (frame["context_bar_open_time"] <= val_end)
            ].copy()
        if fold_active_rows.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{len(folds)} reason=no_active_rows",
            )
            continue
        fold_active_rows = fold_active_rows[
            _build_active_eligibility_mask(
                fold_active_rows,
                resolver_config=resolver_config,
                active_window_end=val_end,
            )
        ].copy()
        if fold_active_rows.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{len(folds)} reason=no_active_rows_after_horizon_filter",
            )
            continue
        fold_warmup_rows = frame[
            (frame["dataset_split"] == "train")
            & (frame["context_bar_open_time"] >= warmup_start)
            & (frame["context_bar_open_time"] < val_start)
            ].copy()
        fold_warmup_rows["policy_context_only"] = True
        fold_active_rows["policy_context_only"] = False
        fold_score_rows = pd.concat(
            [fold_warmup_rows, fold_active_rows], ignore_index=True
        )
        fold_score_rows = fold_score_rows.drop_duplicates(
            subset=["decision_row_id"], keep="last"
        ).reset_index(drop=True)
        scored_fold = _score_policy_window(
            model=model,
            rows_to_score=fold_score_rows,
            score_source="train_oof",
            fold_id=fold.fold_id,
        )
        context_rows = (
            int(scored_fold["policy_context_only"].sum()) if len(scored_fold) > 0 else 0
        )
        active_rows = (
            int((~scored_fold["policy_context_only"]).sum()) if len(scored_fold) > 0 else 0
        )
        log_info(
            "POLICY",
            (
                f"oof fold policy_rows done fold={idx}/{len(folds)} "
                f"rows_total={len(scored_fold)} context_rows={context_rows} active_rows={active_rows} "
                f"elapsed_sec_score={time.perf_counter() - fit_started:.3f}"
            ),
        )
        if scored_fold["decision_row_id"].duplicated().any():
            duplicates = (
                scored_fold.loc[
                    scored_fold["decision_row_id"].duplicated(), "decision_row_id"
                ]
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"fold scoring rows contain duplicate decision_row_id fold_id={fold.fold_id} sample={duplicates}"
            )
        chunks.append(scored_fold)
        elapsed = time.perf_counter() - started
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(folds) - idx) / rate if rate > 0 else 0.0
        rows_accumulated = int(sum(len(chunk) for chunk in chunks))
        log_info(
            "POLICY",
            (
                f"oof progress folds_done={idx}/{len(folds)} rows_accumulated={rows_accumulated} "
                f"elapsed_sec={elapsed:.3f} eta_sec={eta:.3f} fold_elapsed_sec={time.perf_counter() - fold_started:.3f}"
            ),
        )
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        out = out.sort_values(
            ["fold_id", "episode_id", "context_bar_open_time"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
    log_info(
        "POLICY",
        (
            f"policy OOF scoring rows build done rows_total={len(out)} folds_total={len(folds)} "
            f"elapsed_sec_total={time.perf_counter() - started:.3f} "
            f"rows_per_sec={(len(out) / max(time.perf_counter() - started, 1e-9)):.3f}"
        ),
    )
    return out.loc[:, list(POLICY_ROW_COLUMNS)].copy()


def build_detector_policy_grid(
        base_policy_config: DetectorPolicyConfig,
        search_config: DetectorPolicySearchConfig | None = None,
) -> list[DetectorPolicyConfig]:
    if (
            search_config is not None
            and len(search_config.arm_candidates) > 0
            and len(search_config.fire_candidates) > 0
            and len(search_config.turn_candidates) > 0
    ):
        too_early_candidates = (
            list(search_config.max_too_early_candidates)
            if len(search_config.max_too_early_candidates) > 0
            else [base_policy_config.max_too_early_prob]
        )
        too_late_candidates = (
            list(search_config.max_too_late_candidates)
            if len(search_config.max_too_late_candidates) > 0
            else [base_policy_config.max_too_late_prob]
        )
        grid: list[DetectorPolicyConfig] = []
        for (
                arm_score_min,
                fire_score_floor,
                turn_down_delta,
                max_too_early_prob,
                max_too_late_prob,
        ) in product(
                search_config.arm_candidates,
                search_config.fire_candidates,
                search_config.turn_candidates,
                too_early_candidates,
                too_late_candidates,
        ):
            if not (0.0 < float(arm_score_min) <= 1.0):
                continue
            if not (0.0 <= float(fire_score_floor) <= float(arm_score_min)):
                continue
            if not (0.0 < float(turn_down_delta) <= 1.0):
                continue
            if not (0.0 <= float(max_too_early_prob) <= 1.0):
                continue
            if not (0.0 <= float(max_too_late_prob) <= 1.0):
                continue
            grid.append(
                DetectorPolicyConfig(
                    arm_score_min=_round6(float(arm_score_min)),
                    fire_score_floor=_round6(float(fire_score_floor)),
                    turn_down_delta=_round6(float(turn_down_delta)),
                    max_too_early_prob=_round6(float(max_too_early_prob)),
                    max_too_late_prob=_round6(float(max_too_late_prob)),
                )
            )
        unique: dict[tuple[float, float, float, float, float], DetectorPolicyConfig] = {}
        for candidate in grid:
            key = (
                candidate.arm_score_min,
                candidate.fire_score_floor,
                candidate.turn_down_delta,
                candidate.max_too_early_prob,
                candidate.max_too_late_prob,
            )
            unique[key] = candidate
        return sorted(
            unique.values(),
            key=lambda item: (
                item.arm_score_min,
                item.fire_score_floor,
                item.turn_down_delta,
                item.max_too_early_prob,
                item.max_too_late_prob,
            ),
        )
    arm_candidates = sorted(
        {
            _round6(_clip(base_policy_config.arm_score_min + delta, 0.30, 0.95))
            for delta in (-0.10, -0.05, 0.0, 0.05)
        }
    )
    fire_candidates = sorted(
        {
            _round6(_clip(base_policy_config.fire_score_floor + delta, 0.00, 0.90))
            for delta in (-0.10, -0.05, 0.0)
        }
    )
    turn_candidates = sorted(
        {
            _round6(candidate)
            for candidate in (
            base_policy_config.turn_down_delta * 0.5,
            base_policy_config.turn_down_delta,
            base_policy_config.turn_down_delta * 1.5,
            base_policy_config.turn_down_delta * 2.0,
        )
            if candidate > 0.0
        }
    )
    max_too_early_candidates = sorted(
        {
            _round6(_clip(base_policy_config.max_too_early_prob + delta, 0.05, 1.0))
            for delta in (-0.15, 0.0, 0.15)
        }
    )
    max_too_late_candidates = sorted(
        {
            _round6(_clip(base_policy_config.max_too_late_prob + delta, 0.05, 1.0))
            for delta in (-0.15, 0.0, 0.15)
        }
    )
    grid: list[DetectorPolicyConfig] = []
    for arm_score_min, max_too_early_prob, max_too_late_prob in product(
            arm_candidates, max_too_early_candidates, max_too_late_candidates
    ):
        for fire_score_floor, turn_down_delta in product(
                fire_candidates,
                turn_candidates,
        ):
            if fire_score_floor > arm_score_min:
                continue
            if not (0.0 < arm_score_min <= 1.0):
                continue
            if not (0.0 <= fire_score_floor <= arm_score_min):
                continue
            if not (0.0 < turn_down_delta <= 1.0):
                continue
            grid.append(
                DetectorPolicyConfig(
                    arm_score_min=arm_score_min,
                    fire_score_floor=fire_score_floor,
                    turn_down_delta=turn_down_delta,
                    max_too_early_prob=max_too_early_prob,
                    max_too_late_prob=max_too_late_prob,
                )
            )
    return grid


def sweep_detector_policy(
        scored_rows_df: pd.DataFrame,
        base_policy_config: DetectorPolicyConfig,
        search_config: DetectorPolicySearchConfig | None = None,
        window_start: pd.Timestamp | None = None,
        window_end: pd.Timestamp | None = None,
        window_days: float | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    candidates = build_detector_policy_grid(base_policy_config, search_config)
    log_info(
        "POLICY",
        (
            f"policy sweep start candidates_total={len(candidates)} "
            f"episodes_total={scored_rows_df['episode_id'].nunique() if not scored_rows_df.empty else 0} "
            f"active_rows={int((~scored_rows_df['policy_context_only'].astype(bool)).sum()) if 'policy_context_only' in scored_rows_df.columns else len(scored_rows_df)}"
        ),
    )
    rows: list[dict[str, float]] = []
    best_selection_score = float("-inf")
    for idx, candidate in enumerate(candidates, start=1):
        candidate_signals_df, episode_policy_summary_df = (
            apply_episode_aware_detector_policy(
                scored_rows_df, candidate, emit_summary_log=False
            )
        )
        metrics = build_detector_policy_metrics(
            candidate_signals_df,
            episode_policy_summary_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        best_selection_score = max(
            best_selection_score, float(metrics["selection_score"])
        )
        elapsed = time.perf_counter() - started
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(candidates) - idx) / rate if rate > 0 else 0.0
        should_log_progress = idx == 1 or idx == len(candidates) or (idx % 4 == 0)
        if should_log_progress:
            log_info(
                "POLICY",
                (
                    f"policy sweep progress candidate={idx}/{len(candidates)} "
                    f"arm={candidate.arm_score_min:.6f} fire={candidate.fire_score_floor:.6f} "
                    f"turn={candidate.turn_down_delta:.6f} "
                    f"max_too_early={candidate.max_too_early_prob:.6f} "
                    f"max_too_late={candidate.max_too_late_prob:.6f} "
                    f"signals_total={len(candidate_signals_df)} "
                    f"selection_score={float(metrics['selection_score']):.6f} current_best={best_selection_score:.6f} "
                    f"elapsed_sec={elapsed:.3f} eta_sec={eta:.3f}"
                ),
            )
        rows.append(
            {
                "arm_score_min": candidate.arm_score_min,
                "fire_score_floor": candidate.fire_score_floor,
                "turn_down_delta": candidate.turn_down_delta,
                "max_too_early_prob": candidate.max_too_early_prob,
                "max_too_late_prob": candidate.max_too_late_prob,
                "episodes_total": metrics["episodes_total"],
                "episodes_with_good_zone": metrics["episodes_with_good_zone"],
                "episodes_fired": metrics["episodes_fired"],
                "good_episode_capture_rate": metrics["good_episode_capture_rate"],
                "bad_episode_fire_rate": metrics["bad_episode_fire_rate"],
                "fired_good_rate": metrics["fired_good_rate"],
                "fires_per_30d": metrics["fires_per_30d"],
                "median_bars_fire_to_ideal": metrics["median_bars_fire_to_ideal"],
                "median_future_net_edge_pct_at_fire": metrics[
                    "median_future_net_edge_pct_at_fire"
                ],
                "reset_without_fire_share": metrics["reset_without_fire_share"],
                "arm_to_fire_conversion": metrics["arm_to_fire_conversion"],
                "density_sanity_penalty": _compute_detector_density_sanity_penalty(
                    metrics["fires_per_30d"]
                ),
                "selection_score": metrics["selection_score"],
            }
        )
    sweep_df = pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
    log_info(
        "POLICY",
        f"policy sweep done candidates_total={len(sweep_df)} elapsed_sec={time.perf_counter() - started:.3f}",
    )
    return sweep_df


def select_detector_policy(
        scored_rows_df: pd.DataFrame,
        base_policy_config: DetectorPolicyConfig,
        search_config: DetectorPolicySearchConfig | None = None,
        window_start: pd.Timestamp | None = None,
        window_end: pd.Timestamp | None = None,
        window_days: float | None = None,
) -> tuple[DetectorPolicyConfig, pd.DataFrame]:
    sweep_df = sweep_detector_policy(
        scored_rows_df,
        base_policy_config,
        search_config=search_config,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
    )
    if sweep_df.empty:
        raise ValueError("policy sweep returned no candidates")
    ranked = sweep_df.copy()
    positive_fire_exists = bool(
        (
                pd.to_numeric(ranked["episodes_fired"], errors="coerce").fillna(0.0) > 0.0
        ).any()
    )
    if positive_fire_exists:
        ranked = ranked[
            pd.to_numeric(ranked["episodes_fired"], errors="coerce").fillna(0.0) > 0.0
            ].copy()
    ranked["_median_future_edge_sort"] = pd.to_numeric(
        ranked["median_future_net_edge_pct_at_fire"], errors="coerce"
    ).fillna(float("-inf"))
    ranked["_median_bars_abs_sort"] = (
        pd.to_numeric(ranked["median_bars_fire_to_ideal"], errors="coerce")
        .abs()
        .fillna(float("inf"))
    )
    ranked = ranked.sort_values(
        by=[
            "selection_score",
            "_median_future_edge_sort",
            "episodes_fired",
            "_median_bars_abs_sort",
            "arm_score_min",
            "turn_down_delta",
            "max_too_early_prob",
            "max_too_late_prob",
        ],
        ascending=[False, False, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    best_policy = DetectorPolicyConfig(
        arm_score_min=float(best["arm_score_min"]),
        fire_score_floor=float(best["fire_score_floor"]),
        turn_down_delta=float(best["turn_down_delta"]),
        max_too_early_prob=float(best["max_too_early_prob"]),
        max_too_late_prob=float(best["max_too_late_prob"]),
    )
    log_info(
        "POLICY",
        (
            "policy select done "
            f"best_policy=arm={best_policy.arm_score_min:.6f},"
            f"fire={best_policy.fire_score_floor:.6f},"
            f"turn={best_policy.turn_down_delta:.6f},"
            f"max_too_early={best_policy.max_too_early_prob:.6f},"
            f"max_too_late={best_policy.max_too_late_prob:.6f} "
            f"best_selection_score={float(best['selection_score']):.6f}"
        ),
    )
    return best_policy, sweep_df


def build_detector_val_candidate_signal_ledger(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model, scored_rows_df = build_detector_val_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        event_opener_config=event_opener_config,
        detector_model_config=detector_model_config,
    )
    candidate_signals_df, episode_policy_summary_df = (
        apply_episode_aware_detector_policy(
            scored_rows_df=scored_rows_df,
            detector_policy_config=detector_policy_config,
        )
    )
    metrics = build_detector_policy_metrics(
        candidate_signals_df,
        episode_policy_summary_df,
        window_start=pd.Timestamp(split_bounds.train_end),
        window_end=pd.Timestamp(split_bounds.val_end),
    )
    return model, candidate_signals_df, episode_policy_summary_df, metrics


def build_detector_test_candidate_signal_ledger(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
) -> tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model, scored_rows_df = build_detector_test_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        event_opener_config=event_opener_config,
        detector_model_config=detector_model_config,
    )
    candidate_signals_df, episode_policy_summary_df = (
        apply_episode_aware_detector_policy(
            scored_rows_df=scored_rows_df,
            detector_policy_config=detector_policy_config,
        )
    )
    metrics = build_detector_policy_metrics(
        candidate_signals_df,
        episode_policy_summary_df,
        window_start=pd.Timestamp(split_bounds.val_end),
        window_end=pd.Timestamp(split_bounds.test_end),
    )
    log_info(
        "POLICY",
        (
            "detector test candidate ledger done "
            f"episodes_total={int(metrics['episodes_total'])} signals_total={len(candidate_signals_df)}"
        ),
    )
    return model, candidate_signals_df, episode_policy_summary_df, metrics


def build_detector_train_oof_candidate_signal_ledger(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        event_opener_config: EventOpenerConfig,
        detector_cv_config: DetectorCVConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scored_rows_df = build_detector_train_oof_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        event_opener_config=event_opener_config,
        detector_cv_config=detector_cv_config,
        detector_model_config=detector_model_config,
    )
    candidate_signals_df, episode_policy_summary_df = (
        apply_episode_aware_detector_policy(
            scored_rows_df=scored_rows_df,
            detector_policy_config=detector_policy_config,
        )
    )
    train_oof_window_days = compute_eval_window_days_from_policy_rows(scored_rows_df)
    metrics = build_detector_policy_metrics(
        candidate_signals_df,
        episode_policy_summary_df,
        window_days=train_oof_window_days,
    )
    return candidate_signals_df, episode_policy_summary_df, metrics


def _prepare_dataset_frame(dataset_df: pd.DataFrame) -> pd.DataFrame:
    frame = dataset_df.copy()
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
    return frame


def _score_policy_window(
        model: CatBoostClassifier,
        rows_to_score: pd.DataFrame,
        score_source: str,
        fold_id: object,
) -> pd.DataFrame:
    if rows_to_score.empty:
        return pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
    score_frame = predict_detector_scores(
        model, rows_to_score, DETECTOR_FEATURE_COLUMNS
    )
    merged = score_frame.merge(
        rows_to_score[_available_rows_to_score_columns(rows_to_score)],
        on="decision_row_id",
        how="left",
        validate="one_to_one",
    )
    merged["score_source"] = score_source
    merged["fold_id"] = fold_id
    return merged.loc[:, list(POLICY_ROW_COLUMNS)].copy()


def _build_active_eligibility_mask(
        rows_df: pd.DataFrame,
        resolver_config: ResolverConfig,
        active_window_end: pd.Timestamp,
) -> pd.Series:
    if rows_df.empty:
        return pd.Series(dtype=bool)
    horizon_delta = pd.Timedelta(
        minutes=15 * max(int(resolver_config.horizon_bars) - 1, 0)
    )
    horizon_end = (
            pd.to_datetime(rows_df["entry_bar_open_time"], utc=True, errors="raise")
            + horizon_delta
    )
    return horizon_end <= pd.Timestamp(active_window_end)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _round6(value: float) -> float:
    return float(round(float(value), 6))


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")


def _available_rows_to_score_columns(rows_to_score: pd.DataFrame) -> list[str]:
    ordered = [
        "decision_row_id",
        "policy_context_only",
        "episode_age_bars",
        "distance_from_episode_high_pct",
        "episode_runup_from_open_pct",
        "episode_extension_from_open_pct",
        "bars_since_episode_high",
        "drawdown_from_episode_high_so_far",
        "high_retest_count",
        "high_persistence_4",
        "episode_pump_context_streak",
        "target_good_short_now",
        "target_reason",
        "target_reason_group",
        "target_reason_group_id",
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
    ]
    return [column for column in ordered if column in rows_to_score.columns]


def _compute_detector_density_sanity_penalty(fires_per_30d: float) -> float:
    value = float(fires_per_30d)
    if 15.0 <= value <= 180.0:
        return 0.0
    if value < 15.0:
        return float((15.0 - value) / 15.0)
    return float((value - 180.0) / 180.0)
