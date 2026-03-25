from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pump_end_v2.artifacts import ArtifactManager
from pump_end_v2.config import load_and_validate_config
from pump_end_v2.data import (
    build_decision_rows,
    build_episode_summary,
    build_event_quality_report,
    open_causal_pump_episodes,
    prepare_ohlcv_15m_frame,
    resolve_decision_rows,
)
from pump_end_v2.detector import (
    apply_episode_aware_detector_policy,
    assign_detector_dataset_splits,
    build_detector_dataset,
    build_detector_policy_metrics,
    build_detector_target_metrics,
    build_detector_test_policy_rows,
    build_detector_train_oof_candidate_signal_ledger,
    build_detector_val_policy_rows,
    select_detector_policy,
)
from pump_end_v2.execution import (
    build_execution_metrics,
    build_execution_monthly_report,
    build_execution_symbol_report,
    build_execution_window_report,
    prepare_intraday_bars_frame,
    replay_short_signals_with_symbol_lock,
)
from pump_end_v2.features import (
    build_breadth_state_layer,
    build_detector_feature_manifest,
    build_detector_feature_view,
    build_episode_state_layer,
    build_gate_feature_manifest,
    build_reference_state_layer,
    build_token_state_layer,
)
from pump_end_v2.gate import (
    apply_gate_block_threshold,
    build_gate_decile_report,
    build_gate_execution_decision_summary,
    build_gate_test_scored_signals,
    build_gate_val_scored_signals_and_datasets,
    select_gate_block_threshold_execution_aware,
)
from pump_end_v2.logging import log_info, stage_done, stage_start
from pump_end_v2.pipeline.io import (
    load_market_inputs,
    save_dataframe_artifact,
    save_json_artifact,
)
from pump_end_v2.run_context import create_run_context


def run_pump_end_v2_pipeline(config_path: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    stage_start("PIPELINE", "FULL_RUN")
    config = load_and_validate_config(config_path)
    runs_root = config.raw["compute"]["runs_root"]
    run_context = create_run_context(config_path=config_path, runs_root=runs_root)
    manager = ArtifactManager(runs_root)
    config_snapshot_path = manager.save_config_snapshot(run_context.run_dir, config_path)
    run_manifest_path = manager.save_run_manifest(
        run_context.run_dir,
        {
            "run_id": run_context.run_id,
            "mode": "full-run",
            "config_path": str(Path(config_path).resolve()),
            "created_at_utc": run_context.created_at.isoformat(),
            "run_dir": str(run_context.run_dir),
        },
    )
    _log_saved(config_snapshot_path)
    _log_saved(run_manifest_path)

    stage_start("PIPELINE", "INPUT_LOADING")
    raw_15m, raw_1m, raw_1s = load_market_inputs(config)
    bars_15m = prepare_ohlcv_15m_frame(raw_15m)
    bars_1m = prepare_intraday_bars_frame(raw_1m, "1m")
    bars_1s = prepare_intraday_bars_frame(raw_1s, "1s") if raw_1s is not None else None
    stage_done("PIPELINE", "INPUT_LOADING")

    stage_start("PIPELINE", "PREPARED_LAYERS")
    token_state_full = build_token_state_layer(bars_15m, config.event_opener)
    reference_symbols = {config.references.btc_symbol, config.references.eth_symbol}
    token_state_tradable = token_state_full[~token_state_full["symbol"].isin(reference_symbols)].copy()
    reference_state = build_reference_state_layer(
        token_state_full, config.references.btc_symbol, config.references.eth_symbol
    )
    breadth_state = build_breadth_state_layer(token_state_full, config.references.btc_symbol, config.references.eth_symbol)
    stage_done("PIPELINE", "PREPARED_LAYERS")

    stage_start("PIPELINE", "EVENT_CORE")
    episodes = open_causal_pump_episodes(token_state_tradable, config.event_opener)
    decision_rows = build_decision_rows(token_state_tradable, episodes, config.execution)
    resolved_rows = resolve_decision_rows(bars_15m, decision_rows, config.resolver)
    episode_summary = build_episode_summary(resolved_rows)
    event_quality_report = build_event_quality_report(episode_summary)
    stage_done("PIPELINE", "EVENT_CORE")

    stage_start("PIPELINE", "DETECTOR_DATASET")
    episode_state = build_episode_state_layer(token_state_tradable, episodes, decision_rows)
    detector_feature_view = build_detector_feature_view(
        decision_rows,
        token_state_tradable,
        reference_state,
        breadth_state,
        episode_state,
    )
    detector_dataset = build_detector_dataset(detector_feature_view, resolved_rows)
    detector_dataset = assign_detector_dataset_splits(detector_dataset, config.splits)
    stage_done("PIPELINE", "DETECTOR_DATASET")

    stage_start("PIPELINE", "DETECTOR_TRAIN_OOF")
    (
        train_oof_candidate_signals_df,
        train_oof_episode_policy_summary_df,
        detector_train_oof_policy_metrics,
    ) = build_detector_train_oof_candidate_signal_ledger(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_cv_config=config.detector_cv,
        detector_model_config=config.detector_model,
        detector_policy_config=config.detector_policy,
    )
    stage_done("PIPELINE", "DETECTOR_TRAIN_OOF")

    stage_start("PIPELINE", "DETECTOR_VAL_POLICY")
    _, val_policy_rows = build_detector_val_policy_rows(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_model_config=config.detector_model,
    )
    selected_detector_policy, detector_policy_sweep_df = select_detector_policy(
        val_policy_rows, config.detector_policy
    )
    val_candidate_signals_df, val_episode_policy_summary_df = apply_episode_aware_detector_policy(
        val_policy_rows, selected_detector_policy
    )
    detector_val_policy_metrics = build_detector_policy_metrics(
        val_candidate_signals_df,
        val_episode_policy_summary_df,
    )
    detector_val_target_metrics = build_detector_target_metrics(val_policy_rows)
    stage_done("PIPELINE", "DETECTOR_VAL_POLICY")

    stage_start("PIPELINE", "DETECTOR_TEST")
    detector_model_test, test_policy_rows = build_detector_test_policy_rows(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_model_config=config.detector_model,
    )
    test_candidate_signals_df, test_episode_policy_summary_df = apply_episode_aware_detector_policy(
        test_policy_rows, selected_detector_policy
    )
    detector_test_policy_metrics = build_detector_policy_metrics(
        test_candidate_signals_df,
        test_episode_policy_summary_df,
    )
    detector_test_target_metrics = build_detector_target_metrics(test_policy_rows)
    stage_done("PIPELINE", "DETECTOR_TEST")

    stage_start("PIPELINE", "GATE_VAL")
    (
        _,
        val_scored_signals_df,
        gate_threshold_sweep_diagnostic_df,
        gate_dataset_train_oof_df,
        gate_dataset_val_df,
    ) = build_gate_val_scored_signals_and_datasets(
        train_oof_candidate_signals_df,
        val_candidate_signals_df,
        token_state_tradable,
        reference_state,
        breadth_state,
        config.gate_model,
        config.gate_config.block_threshold,
    )
    selected_gate_threshold, gate_threshold_sweep_execution_df = select_gate_block_threshold_execution_aware(
        scored_signals_df=val_scored_signals_df,
        base_block_threshold=config.gate_config.block_threshold,
        bars_1m_df=bars_1m,
        execution_contract=config.execution,
        bars_1s_df=bars_1s,
    )
    val_gate_decisions_df, _ = apply_gate_block_threshold(val_scored_signals_df, selected_gate_threshold)
    stage_done("PIPELINE", "GATE_VAL")

    stage_start("PIPELINE", "GATE_TEST")
    gate_model_test, test_scored_signals_df, gate_dataset_test_df = build_gate_test_scored_signals(
        train_oof_candidate_signals_df,
        test_candidate_signals_df,
        token_state_tradable,
        reference_state,
        breadth_state,
        config.gate_model,
    )
    test_gate_decisions_df, _ = apply_gate_block_threshold(test_scored_signals_df, selected_gate_threshold)
    stage_done("PIPELINE", "GATE_TEST")

    stage_start("PIPELINE", "EXECUTION_REPLAY")
    val_execution_decisions_df, val_executed_signals_df = replay_short_signals_with_symbol_lock(
        val_gate_decisions_df,
        bars_1m,
        config.execution,
        bars_1s,
    )
    test_execution_decisions_df, test_executed_signals_df = replay_short_signals_with_symbol_lock(
        test_gate_decisions_df,
        bars_1m,
        config.execution,
        bars_1s,
    )
    val_decision_summary = build_gate_execution_decision_summary(val_execution_decisions_df)
    test_decision_summary = build_gate_execution_decision_summary(test_execution_decisions_df)
    stage_done("PIPELINE", "EXECUTION_REPLAY")

    stage_start("PIPELINE", "EXECUTION_REPORTS")
    val_metrics = build_execution_metrics(val_executed_signals_df)
    val_window_6h = build_execution_window_report(val_executed_signals_df, 6)
    val_window_24h = build_execution_window_report(val_executed_signals_df, 24)
    val_symbol_report = build_execution_symbol_report(val_executed_signals_df)
    val_monthly_report = build_execution_monthly_report(val_executed_signals_df)
    test_metrics = build_execution_metrics(test_executed_signals_df)
    test_window_6h = build_execution_window_report(test_executed_signals_df, 6)
    test_window_24h = build_execution_window_report(test_executed_signals_df, 24)
    test_symbol_report = build_execution_symbol_report(test_executed_signals_df)
    test_monthly_report = build_execution_monthly_report(test_executed_signals_df)
    gate_deciles_val = build_gate_decile_report(val_scored_signals_df)
    gate_deciles_test = build_gate_decile_report(test_scored_signals_df)
    stage_done("PIPELINE", "EXECUTION_REPORTS")

    stage_start("PIPELINE", "ARTIFACTS_SAVE")
    prepared_dir = manager.stage_output_dir(run_context.run_dir, "prepared")
    detector_dir = manager.stage_output_dir(run_context.run_dir, "detector")
    gate_dir = manager.stage_output_dir(run_context.run_dir, "gate")
    eval_val_dir = manager.stage_output_dir(run_context.run_dir, "eval", "val")
    eval_test_dir = manager.stage_output_dir(run_context.run_dir, "eval", "test")
    reports_dir = manager.stage_output_dir(run_context.run_dir, "reports")

    _save_df_and_log(token_state_full, prepared_dir / "token_state.parquet")
    _save_df_and_log(reference_state, prepared_dir / "reference_state.parquet")
    _save_df_and_log(breadth_state, prepared_dir / "breadth_state.parquet")
    _save_df_and_log(episodes, prepared_dir / "episodes.parquet")
    _save_df_and_log(decision_rows, prepared_dir / "decision_rows.parquet")
    _save_df_and_log(resolved_rows, prepared_dir / "resolved_rows.parquet")
    _save_df_and_log(episode_summary, prepared_dir / "episode_summary.parquet")
    _save_json_and_log(event_quality_report, prepared_dir / "event_quality_report.json")

    _save_df_and_log(detector_dataset, detector_dir / "dataset.parquet")
    _save_json_and_log(build_detector_feature_manifest(), detector_dir / "feature_manifest.json")
    _save_df_and_log(detector_policy_sweep_df, detector_dir / "policy_sweep_val.csv")
    _save_json_and_log(_policy_to_dict(selected_detector_policy), detector_dir / "selected_policy.json")
    _save_df_and_log(train_oof_candidate_signals_df, detector_dir / "train_oof_candidate_signals.parquet")
    _save_df_and_log(
        train_oof_episode_policy_summary_df,
        detector_dir / "train_oof_episode_policy_summary.parquet",
    )
    _save_json_and_log(detector_train_oof_policy_metrics, detector_dir / "train_oof_policy_metrics.json")
    _save_df_and_log(val_policy_rows, detector_dir / "val_policy_rows.parquet")
    _save_df_and_log(val_candidate_signals_df, detector_dir / "val_candidate_signals.parquet")
    _save_df_and_log(val_episode_policy_summary_df, detector_dir / "val_episode_policy_summary.parquet")
    _save_json_and_log(detector_val_policy_metrics, detector_dir / "val_policy_metrics.json")
    _save_json_and_log(detector_val_target_metrics, detector_dir / "val_target_metrics.json")
    _save_df_and_log(test_policy_rows, detector_dir / "test_policy_rows.parquet")
    _save_df_and_log(test_candidate_signals_df, detector_dir / "test_candidate_signals.parquet")
    _save_df_and_log(test_episode_policy_summary_df, detector_dir / "test_episode_policy_summary.parquet")
    _save_json_and_log(detector_test_policy_metrics, detector_dir / "test_policy_metrics.json")
    _save_json_and_log(detector_test_target_metrics, detector_dir / "test_target_metrics.json")
    _save_model_and_log(detector_model_test, detector_dir / "model_train_only.cbm")

    _save_df_and_log(val_scored_signals_df, gate_dir / "val_scored_signals.parquet")
    _save_json_and_log(build_gate_feature_manifest(), gate_dir / "feature_manifest.json")
    _save_df_and_log(gate_dataset_train_oof_df, gate_dir / "dataset_train_oof.parquet")
    _save_df_and_log(gate_dataset_val_df, gate_dir / "dataset_val.parquet")
    _save_df_and_log(gate_dataset_test_df, gate_dir / "dataset_test.parquet")
    _save_df_and_log(gate_threshold_sweep_diagnostic_df, gate_dir / "threshold_sweep_val_diagnostic.csv")
    _save_df_and_log(gate_threshold_sweep_execution_df, gate_dir / "threshold_sweep_val_execution.csv")
    _save_json_and_log(float(selected_gate_threshold), gate_dir / "selected_threshold.json")
    _save_df_and_log(gate_deciles_val, gate_dir / "gate_deciles.csv")
    _save_df_and_log(gate_deciles_test, gate_dir / "gate_deciles_test.csv")
    _save_df_and_log(test_scored_signals_df, gate_dir / "test_scored_signals.parquet")
    _save_model_and_log(gate_model_test, gate_dir / "model_train_oof.cbm")

    _save_df_and_log(val_gate_decisions_df, eval_val_dir / "candidate_signals.parquet")
    _save_df_and_log(val_execution_decisions_df, eval_val_dir / "gate_decisions.parquet")
    _save_df_and_log(val_executed_signals_df, eval_val_dir / "executed_signals.csv")
    _save_json_and_log(val_metrics, eval_val_dir / "metrics.json")
    _save_json_and_log(val_decision_summary, eval_val_dir / "decision_summary.json")
    _save_df_and_log(val_window_6h, eval_val_dir / "window_report_6h.csv")
    _save_df_and_log(val_window_24h, eval_val_dir / "window_report_24h.csv")
    _save_df_and_log(val_symbol_report, eval_val_dir / "symbol_report.csv")
    _save_df_and_log(val_monthly_report, eval_val_dir / "monthly_report.csv")

    _save_df_and_log(test_gate_decisions_df, eval_test_dir / "candidate_signals.parquet")
    holdout_signals_path = _save_df_and_log(test_executed_signals_df, eval_test_dir / "test_signals_holdout.csv")
    _save_df_and_log(test_execution_decisions_df, eval_test_dir / "gate_decisions.parquet")
    _save_json_and_log(test_metrics, eval_test_dir / "metrics_holdout.json")
    _save_json_and_log(test_decision_summary, eval_test_dir / "decision_summary.json")
    _save_df_and_log(test_window_6h, eval_test_dir / "window_report_6h.csv")
    _save_df_and_log(test_window_24h, eval_test_dir / "window_report_24h.csv")
    _save_df_and_log(test_symbol_report, eval_test_dir / "symbol_report.csv")
    _save_df_and_log(test_monthly_report, eval_test_dir / "monthly_report.csv")

    run_summary = {
        "run_id": run_context.run_id,
        "run_dir": str(run_context.run_dir),
        "config_path": str(Path(config_path).resolve()),
        "selected_detector_policy": _policy_to_dict(selected_detector_policy),
        "selected_gate_threshold": float(selected_gate_threshold),
        "event_quality_report": event_quality_report,
        "detector_val_policy_metrics": detector_val_policy_metrics,
        "detector_train_oof_policy_metrics": detector_train_oof_policy_metrics,
        "detector_test_policy_metrics": detector_test_policy_metrics,
        "val_decision_summary": val_decision_summary,
        "test_decision_summary": test_decision_summary,
        "val_execution_metrics": val_metrics,
        "test_execution_metrics": test_metrics,
    }
    summary_path = _save_json_and_log(run_summary, reports_dir / "run_summary.json")
    stage_done("PIPELINE", "ARTIFACTS_SAVE")
    stage_done("PIPELINE", "FULL_RUN", elapsed_sec=time.perf_counter() - started)
    log_info(
        "RUN",
        f"completed run_dir={run_context.run_dir} test_holdout_signals={holdout_signals_path}",
    )
    return {
        "run_id": run_context.run_id,
        "run_dir": str(run_context.run_dir),
        "holdout_signals_path": str(holdout_signals_path),
        "summary_path": str(summary_path),
    }


def _save_df_and_log(df, path: Path) -> Path:
    saved_path = save_dataframe_artifact(df, path)
    _log_saved(saved_path)
    return saved_path


def _save_json_and_log(payload: Any, path: Path) -> Path:
    saved_path = save_json_artifact(payload, path)
    _log_saved(saved_path)
    return saved_path


def _save_model_and_log(model: Any, path: Path) -> Path:
    if not hasattr(model, "save_model"):
        raise TypeError(f"model object does not support save_model: {type(model)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    _log_saved(path)
    return path


def _policy_to_dict(policy: Any) -> dict[str, float]:
    return {
        "arm_score_min": float(policy.arm_score_min),
        "fire_score_floor": float(policy.fire_score_floor),
        "turn_down_delta": float(policy.turn_down_delta),
    }


def _log_saved(path: Path) -> None:
    log_info("ARTIFACTS", f"saved path={path}")
