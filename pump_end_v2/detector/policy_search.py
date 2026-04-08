import math
from itertools import product
import time

import numpy as np
import pandas as pd

from pump_end_v2.config import (
    DetectorCVConfig,
    DetectorModelConfig,
    DetectorPolicyConfig,
    DetectorPolicySearchConfig,
    EventOpenerConfig,
    ResolverConfig,
    SplitBounds,
)
from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.detector.model import (
    SequenceDetector,
    build_sequence_permutation_importance_table,
    build_detector_model,
    fit_detector_model,
    predict_detector_scores,
)
from pump_end_v2.detector.sequence_dataset import DetectorSequenceStore
from pump_end_v2.detector.policy import (
    apply_episode_aware_detector_policy,
    apply_episode_aware_detector_policy_cached,
    build_detector_policy_runtime_cache,
    build_detector_policy_metrics,
    compute_eval_window_days_from_policy_rows,
)
from pump_end_v2.detector.splits import (
    generate_detector_walkforward_folds,
)
from pump_end_v2.features.manifest import DETECTOR_SEQUENCE_FEATURE_COLUMNS
from pump_end_v2.logging import log_info

POLICY_ROW_COLUMNS: tuple[str, ...] = (
    "decision_row_id",
    "episode_id",
    "symbol",
    "context_bar_open_time",
    "decision_time",
    "entry_bar_open_time",
    "p_good",
    "p_tp_row",
    "p_timeout_row",
    "p_sl_row",
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
    "row_trade_outcome",
    "row_trade_pnl_pct",
    "row_mfe_pct",
    "row_mae_pct",
    "row_holding_bars",
    "target_row_weight",
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
    "episodes_total",
    "episodes_with_good_zone",
    "episodes_fired",
    "good_episode_capture_rate",
    "bad_episode_fire_rate",
    "fired_good_rate",
    "fires_per_30d",
    "median_bars_fire_to_ideal",
    "median_row_trade_pnl_pct_at_fire",
    "median_row_mae_pct_at_fire",
    "reset_without_fire_share",
    "arm_to_fire_conversion",
    "density_sanity_penalty",
    "selection_score",
    "resolved_signals_total",
    "tp_rate_resolved",
    "sl_rate_resolved",
    "tp_rate_breakeven",
    "tp_rate_edge_vs_breakeven",
    "edge_vs_breakeven_zscore",
    "selector_density_ok",
    "selector_support_ok",
    "selector_density_distance",
    "timeout_total",
    "timeout_share",
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
)

OOF_IMPORTANCE_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "feature",
    "importance_raw",
    "importance_norm",
)

OOF_SEQUENCE_HISTORY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "epoch",
    "train_loss",
    "eval_loss",
    "is_best_epoch",
    "monitor_name",
)


def build_detector_val_policy_rows(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[SequenceDetector, pd.DataFrame]:
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy val scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    label_horizon_bars = int(execution_contract.entry_shift_bars) + int(
        execution_contract.max_hold_bars
    )
    purge_gap_timedelta = pd.Timedelta(minutes=15 * label_horizon_bars)
    effective_train_end = pd.Timestamp(split_bounds.train_end) - purge_gap_timedelta
    train_fit = frame[
        (frame["dataset_split"] == "train")
        & (frame["context_bar_open_time"] <= effective_train_end)
        ].copy()
    train_inner_fit, fit_eval_df, fit_split_meta = _split_fit_train_eval_chronological(
        train_fit,
        detector_model_config,
        target_column="target_good_short_now",
        purge_gap_timedelta=purge_gap_timedelta,
    )
    if train_inner_fit.empty:
        raise ValueError(
            "no trainable train rows available for detector val policy scoring"
        )
    log_info(
        "POLICY",
        (
            "detector val fit split "
            f"monitor_name={fit_split_meta['monitor_name']} "
            f"train_rows={fit_split_meta['train_rows']} eval_rows={fit_split_meta['eval_rows']} "
            f"fallback_reason={fit_split_meta['fallback_reason']}"
        ),
    )
    model = build_detector_model(detector_model_config)
    fit_detector_model(
        model,
        train_inner_fit,
        DETECTOR_SEQUENCE_FEATURE_COLUMNS,
        "target_good_short_now",
        eval_df=fit_eval_df,
        sequence_store=sequence_store,
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
            execution_contract=execution_contract,
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
        sequence_store=sequence_store,
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
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[SequenceDetector, pd.DataFrame]:
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy test scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    label_horizon_bars = int(execution_contract.entry_shift_bars) + int(
        execution_contract.max_hold_bars
    )
    purge_gap_timedelta = pd.Timedelta(minutes=15 * label_horizon_bars)
    effective_train_end = pd.Timestamp(split_bounds.val_end) - purge_gap_timedelta
    train_fit = frame[
        frame["dataset_split"].isin(["train", "val"])
        & (frame["context_bar_open_time"] <= effective_train_end)
        ].copy()
    train_inner_fit, fit_eval_df, fit_split_meta = _split_fit_train_eval_chronological(
        train_fit,
        detector_model_config,
        target_column="target_good_short_now",
        purge_gap_timedelta=purge_gap_timedelta,
    )
    if train_inner_fit.empty:
        raise ValueError(
            "no trainable train/val rows available for detector test policy scoring"
        )
    log_info(
        "POLICY",
        (
            "detector test fit split "
            f"monitor_name={fit_split_meta['monitor_name']} "
            f"train_rows={fit_split_meta['train_rows']} eval_rows={fit_split_meta['eval_rows']} "
            f"fallback_reason={fit_split_meta['fallback_reason']}"
        ),
    )
    model = build_detector_model(detector_model_config)
    fit_detector_model(
        model,
        train_inner_fit,
        DETECTOR_SEQUENCE_FEATURE_COLUMNS,
        "target_good_short_now",
        eval_df=fit_eval_df,
        sequence_store=sequence_store,
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
            execution_contract=execution_contract,
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
        sequence_store=sequence_store,
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
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_cv_config: DetectorCVConfig,
        detector_model_config: DetectorModelConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    _require_columns(dataset_df, POLICY_BASE_REQUIRED_COLUMNS)
    log_info("POLICY", "policy OOF scoring rows build start")
    frame = _prepare_dataset_frame(dataset_df)
    train_split = frame[frame["dataset_split"] == "train"].copy()
    train_context_ns = train_split["context_bar_open_time"].to_numpy(dtype=np.int64, copy=False)
    folds = generate_detector_walkforward_folds(
        frame, split_bounds, resolver_config, execution_contract, detector_cv_config
    )
    folds_total = len(folds)
    warmup_timedelta = pd.Timedelta(minutes=15 * event_opener_config.max_episode_bars)
    label_horizon_bars = int(execution_contract.entry_shift_bars) + int(
        execution_contract.max_hold_bars
    )
    purge_gap_timedelta = pd.Timedelta(minutes=15 * label_horizon_bars)
    chunks: list[pd.DataFrame] = []
    importance_chunks: list[pd.DataFrame] = []
    history_chunks: list[pd.DataFrame] = []
    rows_accumulated = 0
    for idx, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        log_info(
            "POLICY",
            (
                f"oof fold start fold={idx}/{folds_total} fold_id={fold.fold_id} "
                f"train_rows={fold.train_row_count} val_rows={fold.val_row_count} "
                f"train_start={pd.Timestamp(fold.train_start)} train_end={pd.Timestamp(fold.train_end)} "
                f"val_start={pd.Timestamp(fold.val_start)} val_end={pd.Timestamp(fold.val_end)}"
            ),
        )
        fold_train_start_ns = pd.Timestamp(fold.train_start).value
        fold_train_end_ns = pd.Timestamp(fold.train_end).value
        fold_train_mask = (train_context_ns >= fold_train_start_ns) & (
                train_context_ns <= fold_train_end_ns
        )
        fold_train_raw = train_split.loc[fold_train_mask].copy()
        fold_train_fit, fold_eval_fit, fit_split_meta = _split_fit_train_eval_chronological(
            fold_train_raw,
            detector_model_config,
            target_column="target_good_short_now",
            purge_gap_timedelta=purge_gap_timedelta,
        )
        if fold_train_fit.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{folds_total} reason=no_trainable_rows",
            )
            continue
        model = build_detector_model(detector_model_config)
        fit_started = time.perf_counter()
        fit_detector_model(
            model,
            fold_train_fit,
            DETECTOR_SEQUENCE_FEATURE_COLUMNS,
            "target_good_short_now",
            eval_df=fold_eval_fit,
            sequence_store=sequence_store,
        )
        if getattr(model, "training_history", None):
            fold_history = pd.DataFrame(model.training_history)
            if not fold_history.empty:
                fold_history["fold_id"] = fold.fold_id
                fold_history["monitor_name"] = str(
                    model.train_stats.get("monitor_name", "train_loss_fallback")
                )
                history_chunks.append(
                    fold_history.loc[:, list(OOF_SEQUENCE_HISTORY_COLUMNS)].copy()
                )
        log_info(
            "POLICY",
            (
                f"oof fold model_fit done fold={idx}/{folds_total} "
                f"rows_fit={len(fold_train_fit)} eval_rows={0 if fold_eval_fit is None else len(fold_eval_fit)} "
                f"monitor_name={fit_split_meta['monitor_name']} "
                f"fallback_reason={fit_split_meta['fallback_reason']} "
                f"elapsed_sec_fit={time.perf_counter() - fit_started:.3f}"
            ),
        )
        val_start = pd.Timestamp(fold.val_start)
        val_end = pd.Timestamp(fold.val_end)
        warmup_start = val_start - warmup_timedelta
        val_start_ns = val_start.value
        val_end_ns = val_end.value
        active_mask = (train_context_ns >= val_start_ns) & (
                train_context_ns <= val_end_ns
        )
        fold_active_rows = train_split.loc[active_mask].copy()
        if fold_active_rows.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{folds_total} reason=no_active_rows",
            )
            continue
        fold_active_rows = fold_active_rows[
            _build_active_eligibility_mask(
                fold_active_rows,
                execution_contract=execution_contract,
                active_window_end=val_end,
            )
        ].copy()
        if fold_active_rows.empty:
            log_info(
                "POLICY",
                f"oof fold skipped fold={idx}/{folds_total} reason=no_active_rows_after_horizon_filter",
            )
            continue
        fold_importance_rows = fold_active_rows[
            fold_active_rows["trainable_row"].astype(bool)
        ].copy()
        if not fold_importance_rows.empty:
            fold_importance = build_sequence_permutation_importance_table(
                model=model,
                eval_df=fold_importance_rows,
                target_column="target_good_short_now",
                sequence_store=sequence_store,
            ).copy()
            fold_importance["fold_id"] = fold.fold_id
            importance_chunks.append(
                fold_importance.loc[:, list(OOF_IMPORTANCE_COLUMNS)].copy()
            )
        warmup_start_ns = warmup_start.value
        warmup_mask = (train_context_ns >= warmup_start_ns) & (
                train_context_ns < val_start_ns
        )
        fold_warmup_rows = train_split.loc[warmup_mask].copy()
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
            sequence_store=sequence_store,
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
                f"oof fold policy_rows done fold={idx}/{folds_total} "
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
        rows_accumulated += len(scored_fold)
        elapsed = time.perf_counter() - started
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (folds_total - idx) / rate if rate > 0 else 0.0
        log_info(
            "POLICY",
            (
                f"oof progress folds_done={idx}/{folds_total} rows_accumulated={rows_accumulated} "
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
    if importance_chunks:
        oof_importance_df = pd.concat(importance_chunks, ignore_index=True)
        oof_importance_df = oof_importance_df.loc[:, list(OOF_IMPORTANCE_COLUMNS)].copy()
    else:
        oof_importance_df = pd.DataFrame(columns=list(OOF_IMPORTANCE_COLUMNS))
    if history_chunks:
        oof_history_df = pd.concat(history_chunks, ignore_index=True)
        oof_history_df = oof_history_df.loc[:, list(OOF_SEQUENCE_HISTORY_COLUMNS)].copy()
    else:
        oof_history_df = pd.DataFrame(columns=list(OOF_SEQUENCE_HISTORY_COLUMNS))
    log_info(
        "POLICY",
        (
            f"policy OOF scoring rows build done rows_total={len(out)} folds_total={folds_total} "
            f"elapsed_sec_total={time.perf_counter() - started:.3f} "
            f"rows_per_sec={(len(out) / max(time.perf_counter() - started, 1e-9)):.3f}"
        ),
    )
    return out.loc[:, list(POLICY_ROW_COLUMNS)].copy(), oof_importance_df, oof_history_df


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
        grid: list[DetectorPolicyConfig] = []
        for arm_score_min, fire_score_floor, turn_down_delta in product(
                search_config.arm_candidates,
                search_config.fire_candidates,
                search_config.turn_candidates,
        ):
            if not (0.0 < float(arm_score_min) <= 1.0):
                continue
            if not (0.0 <= float(fire_score_floor) <= float(arm_score_min)):
                continue
            if not (0.0 < float(turn_down_delta) <= 1.0):
                continue
            grid.append(
                DetectorPolicyConfig(
                    arm_score_min=_round6(float(arm_score_min)),
                    fire_score_floor=_round6(float(fire_score_floor)),
                    turn_down_delta=_round6(float(turn_down_delta)),
                )
            )
        unique: dict[tuple[float, float, float], DetectorPolicyConfig] = {}
        for candidate in grid:
            key = (
                candidate.arm_score_min,
                candidate.fire_score_floor,
                candidate.turn_down_delta,
            )
            unique[key] = candidate
        return sorted(
            unique.values(),
            key=lambda item: (
                item.arm_score_min,
                item.fire_score_floor,
                item.turn_down_delta,
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
    grid: list[DetectorPolicyConfig] = []
    for arm_score_min in arm_candidates:
        for fire_score_floor, turn_down_delta in product(
                fire_candidates, turn_candidates
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
                )
            )
    return grid


def sweep_detector_policy(
        scored_rows_df: pd.DataFrame,
        base_policy_config: DetectorPolicyConfig,
        execution_contract: ExecutionContract,
        search_config: DetectorPolicySearchConfig | None = None,
        window_start: pd.Timestamp | None = None,
        window_end: pd.Timestamp | None = None,
        window_days: float | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    candidates = build_detector_policy_grid(base_policy_config, search_config)
    runtime_cache = build_detector_policy_runtime_cache(scored_rows_df)
    (
        selector_val_fires_per_30d_min,
        selector_val_fires_per_30d_max,
        selector_min_resolved_signals,
    ) = _resolve_selector_thresholds(search_config)
    density_center = (
                             selector_val_fires_per_30d_min + selector_val_fires_per_30d_max
                     ) / 2.0
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
            apply_episode_aware_detector_policy_cached(
                runtime_cache=runtime_cache,
                detector_policy_config=candidate,
                emit_summary_log=False,
            )
        )
        metrics = build_detector_policy_metrics(
            candidate_signals_df,
            episode_policy_summary_df,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        selector_metrics = _build_execution_aware_selector_metrics(
            candidate_signals_df, execution_contract
        )
        fires_per_30d = float(metrics["fires_per_30d"])
        resolved_signals_total = float(selector_metrics["resolved_signals_total"])
        selector_density_ok = (
                selector_val_fires_per_30d_min
                <= fires_per_30d
                <= selector_val_fires_per_30d_max
        )
        selector_support_ok = (
                resolved_signals_total >= float(selector_min_resolved_signals)
        )
        selector_density_distance = abs(fires_per_30d - density_center)
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
                    f"turn={candidate.turn_down_delta:.6f} signals_total={len(candidate_signals_df)} "
                    f"tp_rate_resolved={float(selector_metrics['tp_rate_resolved']):.6f} "
                    f"resolved_signals_total={int(round(resolved_signals_total))} "
                    f"timeout_total={int(round(float(selector_metrics['timeout_total'])))} "
                    f"timeout_share={float(selector_metrics['timeout_share']):.6f} "
                    f"selection_score={float(metrics['selection_score']):.6f} current_best={best_selection_score:.6f} "
                    f"elapsed_sec={elapsed:.3f} eta_sec={eta:.3f}"
                ),
            )
        rows.append(
            {
                "arm_score_min": candidate.arm_score_min,
                "fire_score_floor": candidate.fire_score_floor,
                "turn_down_delta": candidate.turn_down_delta,
                "episodes_total": metrics["episodes_total"],
                "episodes_with_good_zone": metrics["episodes_with_good_zone"],
                "episodes_fired": metrics["episodes_fired"],
                "good_episode_capture_rate": metrics["good_episode_capture_rate"],
                "bad_episode_fire_rate": metrics["bad_episode_fire_rate"],
                "fired_good_rate": metrics["fired_good_rate"],
                "fires_per_30d": metrics["fires_per_30d"],
                "median_bars_fire_to_ideal": metrics["median_bars_fire_to_ideal"],
                "median_row_trade_pnl_pct_at_fire": metrics[
                    "median_row_trade_pnl_pct_at_fire"
                ],
                "median_row_mae_pct_at_fire": metrics["median_row_mae_pct_at_fire"],
                "reset_without_fire_share": metrics["reset_without_fire_share"],
                "arm_to_fire_conversion": metrics["arm_to_fire_conversion"],
                "density_sanity_penalty": _compute_detector_density_sanity_penalty(
                    metrics["fires_per_30d"]
                ),
                "selection_score": metrics["selection_score"],
                "resolved_signals_total": resolved_signals_total,
                "tp_rate_resolved": selector_metrics["tp_rate_resolved"],
                "sl_rate_resolved": selector_metrics["sl_rate_resolved"],
                "tp_rate_breakeven": selector_metrics["tp_rate_breakeven"],
                "tp_rate_edge_vs_breakeven": selector_metrics["tp_rate_edge_vs_breakeven"],
                "edge_vs_breakeven_zscore": selector_metrics["edge_vs_breakeven_zscore"],
                "selector_density_ok": selector_density_ok,
                "selector_support_ok": selector_support_ok,
                "selector_density_distance": selector_density_distance,
                "timeout_total": selector_metrics["timeout_total"],
                "timeout_share": selector_metrics["timeout_share"],
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
        execution_contract: ExecutionContract,
        search_config: DetectorPolicySearchConfig | None = None,
        window_start: pd.Timestamp | None = None,
        window_end: pd.Timestamp | None = None,
        window_days: float | None = None,
) -> tuple[DetectorPolicyConfig, pd.DataFrame]:
    sweep_df = sweep_detector_policy(
        scored_rows_df,
        base_policy_config,
        execution_contract=execution_contract,
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
    (
        selector_val_fires_per_30d_min,
        selector_val_fires_per_30d_max,
        selector_min_resolved_signals,
    ) = _resolve_selector_thresholds(search_config)
    density_center = (
                             selector_val_fires_per_30d_min + selector_val_fires_per_30d_max
                     ) / 2.0
    fires_per_30d = pd.to_numeric(ranked["fires_per_30d"], errors="coerce").fillna(0.0)
    resolved_signals_total = pd.to_numeric(
        ranked["resolved_signals_total"], errors="coerce"
    ).fillna(0.0)
    ranked["selector_density_ok"] = (
            (fires_per_30d >= selector_val_fires_per_30d_min)
            & (fires_per_30d <= selector_val_fires_per_30d_max)
    ).astype(bool)
    ranked["selector_support_ok"] = (
            resolved_signals_total >= float(selector_min_resolved_signals)
    ).astype(bool)
    ranked["selector_density_distance"] = (fires_per_30d - density_center).abs()
    density_and_support = ranked[
        ranked["selector_density_ok"] & ranked["selector_support_ok"]
        ].copy()
    support_only = ranked[ranked["selector_support_ok"]].copy()
    if not density_and_support.empty:
        admissible = density_and_support
        fallback_tier = "density_and_support"
    elif not support_only.empty:
        admissible = support_only
        fallback_tier = "support_only"
    else:
        admissible = ranked.copy()
        fallback_tier = "positive_fire_only"
    admissible["tp_rate_resolved"] = pd.to_numeric(
        admissible["tp_rate_resolved"], errors="coerce"
    ).fillna(float("-inf"))
    admissible["edge_vs_breakeven_zscore"] = pd.to_numeric(
        admissible["edge_vs_breakeven_zscore"], errors="coerce"
    ).fillna(float("-inf"))
    admissible["selector_density_distance"] = pd.to_numeric(
        admissible["selector_density_distance"], errors="coerce"
    ).fillna(float("inf"))
    admissible["resolved_signals_total"] = pd.to_numeric(
        admissible["resolved_signals_total"], errors="coerce"
    ).fillna(0.0)
    ranked = admissible.sort_values(
        by=[
            "tp_rate_resolved",
            "edge_vs_breakeven_zscore",
            "selector_density_distance",
            "resolved_signals_total",
            "arm_score_min",
            "fire_score_floor",
            "turn_down_delta",
        ],
        ascending=[False, False, True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    best_policy = DetectorPolicyConfig(
        arm_score_min=float(best["arm_score_min"]),
        fire_score_floor=float(best["fire_score_floor"]),
        turn_down_delta=float(best["turn_down_delta"]),
    )
    log_info(
        "POLICY",
        (
            "policy select done "
            f"best_policy=arm={best_policy.arm_score_min:.6f},"
            f"fire={best_policy.fire_score_floor:.6f},"
            f"turn={best_policy.turn_down_delta:.6f} "
            f"tp_rate_resolved={float(best['tp_rate_resolved']):.6f} "
            f"edge_vs_breakeven_zscore={float(best['edge_vs_breakeven_zscore']):.6f} "
            f"fires_per_30d={float(best['fires_per_30d']):.6f} "
            f"resolved_signals_total={int(round(float(best['resolved_signals_total'])))} "
            f"timeout_total={int(round(float(best['timeout_total'])))} "
            f"timeout_share={float(best['timeout_share']):.6f} "
            f"selector_density_ok={bool(best['selector_density_ok'])} "
            f"selector_support_ok={bool(best['selector_support_ok'])} "
            f"fallback_tier={fallback_tier}"
        ),
    )
    return best_policy, sweep_df


def build_detector_val_candidate_signal_ledger(
        dataset_df: pd.DataFrame,
        split_bounds: SplitBounds,
        resolver_config: ResolverConfig,
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[SequenceDetector, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model, scored_rows_df = build_detector_val_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        execution_contract=execution_contract,
        event_opener_config=event_opener_config,
        detector_model_config=detector_model_config,
        sequence_store=sequence_store,
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
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[SequenceDetector, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model, scored_rows_df = build_detector_test_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        execution_contract=execution_contract,
        event_opener_config=event_opener_config,
        detector_model_config=detector_model_config,
        sequence_store=sequence_store,
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
        execution_contract: ExecutionContract,
        event_opener_config: EventOpenerConfig,
        detector_cv_config: DetectorCVConfig,
        detector_model_config: DetectorModelConfig,
        detector_policy_config: DetectorPolicyConfig,
        sequence_store: DetectorSequenceStore,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scored_rows_df, _, _ = build_detector_train_oof_policy_rows(
        dataset_df=dataset_df,
        split_bounds=split_bounds,
        resolver_config=resolver_config,
        execution_contract=execution_contract,
        event_opener_config=event_opener_config,
        detector_cv_config=detector_cv_config,
        detector_model_config=detector_model_config,
        sequence_store=sequence_store,
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


def _split_fit_train_eval_chronological(
        fit_frame: pd.DataFrame,
        detector_model_config: DetectorModelConfig,
        target_column: str,
        purge_gap_timedelta: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    if fit_frame.empty:
        return (
            fit_frame.copy(),
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": 0,
                "eval_rows": 0,
                "fallback_reason": "empty_fit_frame",
            },
        )
    trainable = fit_frame[fit_frame["trainable_row"].astype(bool)].copy()
    if trainable.empty:
        return (
            trainable,
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": 0,
                "eval_rows": 0,
                "fallback_reason": "no_trainable_rows",
            },
        )
    trainable = trainable.sort_values(
        ["context_bar_open_time", "decision_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    total_rows = int(len(trainable))
    eval_rows = int(
        max(
            int(detector_model_config.fit_eval_min_rows),
            int(np.floor(float(total_rows) * float(detector_model_config.fit_eval_fraction))),
        )
    )
    if eval_rows <= 0:
        return (
            trainable,
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": int(len(trainable)),
                "eval_rows": 0,
                "fallback_reason": "eval_rows_non_positive",
            },
        )
    if eval_rows >= total_rows:
        return (
            trainable,
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": int(len(trainable)),
                "eval_rows": 0,
                "fallback_reason": "eval_rows_exhaust_train",
            },
        )
    split_idx = int(total_rows - eval_rows)
    eval_start_time = pd.Timestamp(trainable.iloc[split_idx]["context_bar_open_time"])
    train_cutoff_time = eval_start_time - purge_gap_timedelta
    train_inner = trainable[
        trainable["context_bar_open_time"] < train_cutoff_time
        ].copy()
    eval_inner = trainable[
        trainable["context_bar_open_time"] >= eval_start_time
        ].copy()
    if len(train_inner) < 2 or len(eval_inner) < 2:
        return (
            trainable,
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": int(len(trainable)),
                "eval_rows": 0,
                "fallback_reason": "split_too_small",
            },
        )
    train_single_class = (
            pd.to_numeric(train_inner[target_column], errors="coerce").fillna(0.0).nunique() < 2
    )
    eval_single_class = (
            pd.to_numeric(eval_inner[target_column], errors="coerce").fillna(0.0).nunique() < 2
    )
    if train_single_class or eval_single_class:
        reason = "train_single_class" if train_single_class else "eval_single_class"
        return (
            trainable,
            None,
            {
                "monitor_name": "train_loss_fallback",
                "train_rows": int(len(trainable)),
                "eval_rows": 0,
                "fallback_reason": reason,
            },
        )
    if detector_model_config.main_target_mode == "tp_vs_sl_only":
        train_main_rows = train_inner[
            train_inner["target_reason"].astype(str).str.strip().str.lower().isin(
                ["tp", "sl"]
            )
        ]
        eval_main_rows = eval_inner[
            eval_inner["target_reason"].astype(str).str.strip().str.lower().isin(
                ["tp", "sl"]
            )
        ]
        train_main_single_class = (
                train_main_rows["target_reason"]
                .astype(str)
                .str.strip()
                .str.lower()
                .nunique()
                < 2
        )
        eval_main_single_class = (
                eval_main_rows["target_reason"]
                .astype(str)
                .str.strip()
                .str.lower()
                .nunique()
                < 2
        )
        if train_main_single_class or eval_main_single_class:
            reason = (
                "train_main_single_class"
                if train_main_single_class
                else "eval_main_single_class"
            )
            return (
                trainable,
                None,
                {
                    "monitor_name": "train_loss_fallback",
                    "train_rows": int(len(trainable)),
                    "eval_rows": 0,
                    "fallback_reason": reason,
                },
            )
    return (
        train_inner,
        eval_inner,
        {
            "monitor_name": "eval_loss",
            "train_rows": int(len(train_inner)),
            "eval_rows": int(len(eval_inner)),
            "fallback_reason": "",
        },
    )


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
        model: SequenceDetector,
        rows_to_score: pd.DataFrame,
        score_source: str,
        fold_id: object,
        sequence_store: DetectorSequenceStore,
) -> pd.DataFrame:
    if rows_to_score.empty:
        return pd.DataFrame(columns=list(POLICY_ROW_COLUMNS))
    score_frame = predict_detector_scores(
        model,
        rows_to_score,
        DETECTOR_SEQUENCE_FEATURE_COLUMNS,
        sequence_store=sequence_store,
    )
    merged = score_frame.merge(
        rows_to_score[_available_rows_to_score_columns(rows_to_score)],
        on="decision_row_id",
        how="left",
        validate="one_to_one",
    )
    merged["score_source"] = score_source
    merged["fold_id"] = fold_id
    _log_p_good_distribution(
        merged,
        score_source=score_source,
        fold_id=fold_id,
    )
    return merged.loc[:, list(POLICY_ROW_COLUMNS)].copy()


def _build_active_eligibility_mask(
        rows_df: pd.DataFrame,
        execution_contract: ExecutionContract,
        active_window_end: pd.Timestamp,
) -> pd.Series:
    if rows_df.empty:
        return pd.Series(dtype=bool)
    label_horizon_bars = int(execution_contract.entry_shift_bars) + int(
        execution_contract.max_hold_bars
    )
    horizon_delta = pd.Timedelta(minutes=15 * max(label_horizon_bars - 1, 0))
    entry_bar_open_time = rows_df["entry_bar_open_time"]
    if pd.api.types.is_datetime64_any_dtype(
            entry_bar_open_time
    ) or pd.api.types.is_datetime64tz_dtype(entry_bar_open_time):
        entry_times = entry_bar_open_time
    else:
        entry_times = pd.to_datetime(entry_bar_open_time, utc=True, errors="raise")
    horizon_end = entry_times + horizon_delta
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
        "row_trade_outcome",
        "row_trade_pnl_pct",
        "row_mfe_pct",
        "row_mae_pct",
        "row_holding_bars",
        "target_row_weight",
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


def _resolve_selector_thresholds(
        search_config: DetectorPolicySearchConfig | None,
) -> tuple[float, float, int]:
    if search_config is None:
        return 110.0, 180.0, 80
    return (
        float(search_config.selector_val_fires_per_30d_min),
        float(search_config.selector_val_fires_per_30d_max),
        int(search_config.selector_min_resolved_signals),
    )


def _build_execution_aware_selector_metrics(
        candidate_signals_df: pd.DataFrame, execution_contract: ExecutionContract
) -> dict[str, float | None]:
    if candidate_signals_df.empty or "row_trade_outcome" not in candidate_signals_df.columns:
        resolved_signals_total = 0
        tp_total = 0
        sl_total = 0
        timeout_total = 0
    else:
        outcome = (
            candidate_signals_df["row_trade_outcome"].astype(str).str.strip().str.lower()
        )
        tp_total = int(outcome.eq("tp").sum())
        sl_total = int(outcome.eq("sl").sum())
        timeout_total = int(outcome.eq("timeout").sum())
        resolved_signals_total = int(tp_total + sl_total)
    tp_rate_resolved = (
        float(tp_total / resolved_signals_total)
        if resolved_signals_total > 0
        else 0.0
    )
    sl_rate_resolved = (
        float(sl_total / resolved_signals_total)
        if resolved_signals_total > 0
        else 0.0
    )
    tp_pct = float(execution_contract.tp_pct)
    sl_pct_abs = abs(float(execution_contract.sl_pct))
    breakeven_denominator = tp_pct + sl_pct_abs
    tp_rate_breakeven = (
        float(sl_pct_abs / breakeven_denominator)
        if breakeven_denominator > 0.0
        else 0.5
    )
    tp_rate_edge_vs_breakeven = float(tp_rate_resolved - tp_rate_breakeven)
    if resolved_signals_total <= 0:
        edge_vs_breakeven_zscore = math.nan
    else:
        variance = (
                tp_rate_breakeven
                * (1.0 - tp_rate_breakeven)
                / float(resolved_signals_total)
        )
        edge_vs_breakeven_zscore = (
            float(tp_rate_edge_vs_breakeven / math.sqrt(variance))
            if variance > 0.0
            else math.nan
        )
    return {
        "resolved_signals_total": float(resolved_signals_total),
        "tp_total": float(tp_total),
        "sl_total": float(sl_total),
        "timeout_total": float(timeout_total),
        "timeout_share": float(
            timeout_total / len(candidate_signals_df) if len(candidate_signals_df) > 0 else 0.0
        ),
        "tp_rate_resolved": float(tp_rate_resolved),
        "sl_rate_resolved": float(sl_rate_resolved),
        "tp_rate_breakeven": float(tp_rate_breakeven),
        "tp_rate_edge_vs_breakeven": float(tp_rate_edge_vs_breakeven),
        "edge_vs_breakeven_zscore": float(edge_vs_breakeven_zscore),
    }


def _log_p_good_distribution(
        scored_rows_df: pd.DataFrame,
        score_source: str,
        fold_id: object,
) -> None:
    if scored_rows_df.empty:
        log_info(
            "POLICY",
            (
                f"p_good distribution source={score_source} fold_id={fold_id} "
                "active_rows=0 nan_share=0.000000 min=0.000000 q01=0.000000 "
                "q10=0.000000 q50=0.000000 q90=0.000000 q99=0.000000 max=0.000000"
            ),
        )
        return
    active_mask = ~scored_rows_df["policy_context_only"].astype(bool)
    active_p_good = pd.to_numeric(
        scored_rows_df.loc[active_mask, "p_good"], errors="coerce"
    )
    active_count = int(active_p_good.shape[0])
    if active_count <= 0:
        log_info(
            "POLICY",
            (
                f"p_good distribution source={score_source} fold_id={fold_id} "
                "active_rows=0 nan_share=0.000000 min=0.000000 q01=0.000000 "
                "q10=0.000000 q50=0.000000 q90=0.000000 q99=0.000000 max=0.000000"
            ),
        )
        return
    values = active_p_good.to_numpy(dtype=float, copy=False)
    finite_values = values[np.isfinite(values)]
    nan_share = float(np.mean(~np.isfinite(values)))
    if finite_values.size == 0:
        min_v = 0.0
        q01_v = 0.0
        q10_v = 0.0
        q50_v = 0.0
        q90_v = 0.0
        q99_v = 0.0
        max_v = 0.0
    else:
        min_v = float(np.min(finite_values))
        q01_v = float(np.quantile(finite_values, 0.01))
        q10_v = float(np.quantile(finite_values, 0.10))
        q50_v = float(np.quantile(finite_values, 0.50))
        q90_v = float(np.quantile(finite_values, 0.90))
        q99_v = float(np.quantile(finite_values, 0.99))
        max_v = float(np.max(finite_values))
    log_info(
        "POLICY",
        (
            f"p_good distribution source={score_source} fold_id={fold_id} "
            f"active_rows={active_count} nan_share={nan_share:.6f} min={min_v:.6f} "
            f"q01={q01_v:.6f} q10={q10_v:.6f} q50={q50_v:.6f} q90={q90_v:.6f} "
            f"q99={q99_v:.6f} max={max_v:.6f}"
        ),
    )
