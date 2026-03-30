import time
from pathlib import Path
from typing import Any

import pandas as pd

from pump_end_v2.artifacts import ArtifactManager
from pump_end_v2.config import load_and_validate_config
from pump_end_v2.data import (
    ClickHouseMarketDataLoader,
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
    build_detector_train_oof_policy_rows,
    build_detector_val_policy_rows,
    compute_eval_window_days_from_policy_rows,
    select_detector_policy,
)
from pump_end_v2.execution import (
    build_execution_market_view,
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
    build_candidate_signal_strength_report,
    build_gate_decile_report,
    build_gate_execution_decision_summary,
    build_gate_rank_quality_report,
    build_gate_test_scored_signals,
    build_gate_val_scored_signals_and_datasets,
    select_gate_block_threshold_execution_aware,
)
from pump_end_v2.logging import log_info, stage_done, stage_start
from pump_end_v2.pipeline.io import (
    save_dataframe_artifact,
    save_json_artifact,
)
from pump_end_v2.run_context import create_run_context


def run_pump_end_v2_pipeline(
    config_path: str | Path, clickhouse_dsn: str
) -> dict[str, object]:
    started = time.perf_counter()
    stage_start("PIPELINE", "FULL_RUN")
    config = load_and_validate_config(config_path)
    runs_root = config.raw["compute"]["runs_root"]
    run_context = create_run_context(config_path=config_path, runs_root=runs_root)
    manager = ArtifactManager(runs_root)
    config_snapshot_path = manager.save_config_snapshot(
        run_context.run_dir, config_path
    )
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
    log_info(
        "ARTIFACTS",
        f"run artifacts initialized run_dir={run_context.run_dir}",
    )
    market_loader = ClickHouseMarketDataLoader(clickhouse_dsn, config)
    universe_symbols = market_loader.build_universe_symbols()
    log_info(
        "PIPELINE",
        (
            "FULL_RUN context "
            f"run_id={run_context.run_id} run_dir={run_context.run_dir} "
            f"config_path={Path(config_path).resolve()} "
            f"window_start={pd.Timestamp(config.data_window.start)} "
            f"window_end={pd.Timestamp(config.data_window.end) if config.data_window.end is not None else pd.Timestamp(config.splits.test_end)} "
            f"train_end={pd.Timestamp(config.splits.train_end)} "
            f"val_end={pd.Timestamp(config.splits.val_end)} "
            f"test_end={pd.Timestamp(config.splits.test_end)} "
            f"symbols_total={len(universe_symbols)} "
            f"execution_contract=tp_pct:{float(config.execution.tp_pct):.6f},sl_pct:{float(config.execution.sl_pct):.6f},max_hold_bars:{int(config.execution.max_hold_bars)} "
            f"detector_policy_baseline=arm:{float(config.detector_policy.arm_score_min):.6f},fire:{float(config.detector_policy.fire_score_floor):.6f},turn:{float(config.detector_policy.turn_down_delta):.6f} "
            f"gate_threshold_baseline={float(config.gate_config.block_threshold):.6f}"
        ),
    )
    bars_1s_fetcher = market_loader.build_1s_fetcher()

    input_started = time.perf_counter()
    stage_start("PIPELINE", "INPUT_LOADING")
    raw_15m = market_loader.load_15m_ohlcv()
    bars_15m = prepare_ohlcv_15m_frame(raw_15m)
    input_elapsed = time.perf_counter() - input_started
    rows_total = len(bars_15m)
    symbols_total = int(bars_15m["symbol"].nunique()) if not bars_15m.empty else 0
    open_time_min = (
        pd.Timestamp(bars_15m["open_time"].min()) if not bars_15m.empty else pd.NaT
    )
    open_time_max = (
        pd.Timestamp(bars_15m["open_time"].max()) if not bars_15m.empty else pd.NaT
    )
    rows_per_sec = (rows_total / input_elapsed) if input_elapsed > 0 else 0.0
    log_info(
        "PIPELINE",
        (
            f"INPUT_LOADING summary rows_total={rows_total} symbols_total={symbols_total} "
            f"open_time_min={open_time_min} open_time_max={open_time_max} "
            f"elapsed_sec={input_elapsed:.3f} rows_per_sec={rows_per_sec:.3f}"
        ),
    )
    stage_done("PIPELINE", "INPUT_LOADING", elapsed_sec=input_elapsed)

    prepared_started = time.perf_counter()
    stage_start("PIPELINE", "PREPARED_LAYERS")
    token_state_full = build_token_state_layer(bars_15m, config.event_opener)
    reference_symbols = {config.references.btc_symbol, config.references.eth_symbol}
    token_state_tradable = token_state_full[
        ~token_state_full["symbol"].isin(reference_symbols)
    ].copy()
    reference_state = build_reference_state_layer(
        token_state_full, config.references.btc_symbol, config.references.eth_symbol
    )
    breadth_state = build_breadth_state_layer(
        token_state_full, config.references.btc_symbol, config.references.eth_symbol
    )
    prepared_elapsed = time.perf_counter() - prepared_started
    log_info(
        "PIPELINE",
        (
            f"PREPARED_LAYERS summary elapsed_sec_total={prepared_elapsed:.3f} "
            f"token_rows={len(token_state_tradable)} reference_rows={len(reference_state)} "
            f"breadth_rows={len(breadth_state)}"
        ),
    )
    stage_done("PIPELINE", "PREPARED_LAYERS", elapsed_sec=prepared_elapsed)

    event_started = time.perf_counter()
    stage_start("PIPELINE", "EVENT_CORE")
    episodes = open_causal_pump_episodes(token_state_tradable, config.event_opener)
    decision_rows = build_decision_rows(
        token_state_tradable, episodes, config.execution
    )
    resolved_rows = resolve_decision_rows(bars_15m, decision_rows, config.resolver)
    episode_summary = build_episode_summary(resolved_rows)
    event_quality_report = build_event_quality_report(episode_summary)
    event_elapsed = time.perf_counter() - event_started
    resolved_rows_count = int(resolved_rows["is_resolved"].sum()) if not resolved_rows.empty else 0
    good_rows_count = int((resolved_rows["target_good_short_now"] == 1).sum()) if not resolved_rows.empty else 0
    good_episode_share = float(event_quality_report.get("good_episode_share", 0.0))
    log_info(
        "PIPELINE",
        (
            f"EVENT_CORE summary elapsed_sec_total={event_elapsed:.3f} episodes_total={len(episodes)} "
            f"decision_rows_total={len(decision_rows)} resolved_rows={resolved_rows_count} "
            f"good_rows={good_rows_count} good_episode_share={good_episode_share:.6f}"
        ),
    )
    stage_done("PIPELINE", "EVENT_CORE", elapsed_sec=event_elapsed)

    detector_dataset_started = time.perf_counter()
    stage_start("PIPELINE", "DETECTOR_DATASET")
    episode_state = build_episode_state_layer(
        token_state_tradable, episodes, decision_rows
    )
    detector_feature_view = build_detector_feature_view(
        decision_rows,
        token_state_tradable,
        reference_state,
        breadth_state,
        episode_state,
    )
    detector_dataset = build_detector_dataset(detector_feature_view, resolved_rows)
    detector_dataset = assign_detector_dataset_splits(detector_dataset, config.splits)
    detector_dataset_elapsed = time.perf_counter() - detector_dataset_started
    split_counts = detector_dataset["dataset_split"].value_counts(dropna=False).to_dict()
    positive_rate = (
        float(
            detector_dataset.loc[
                detector_dataset["trainable_row"].astype(bool), "target_good_short_now"
            ]
            .astype(float)
            .mean()
        )
        if bool(detector_dataset["trainable_row"].astype(bool).any())
        else 0.0
    )
    log_info(
        "PIPELINE",
        (
            f"DETECTOR_DATASET summary elapsed_sec_total={detector_dataset_elapsed:.3f} "
            f"rows_total={len(detector_dataset)} train_rows={int(split_counts.get('train', 0))} "
            f"val_rows={int(split_counts.get('val', 0))} test_rows={int(split_counts.get('test', 0))} "
            f"positive_rate={positive_rate:.6f} feature_cols_total={len(build_detector_feature_manifest().get('feature_columns', []))}"
        ),
    )
    stage_done("PIPELINE", "DETECTOR_DATASET", elapsed_sec=detector_dataset_elapsed)

    detector_oof_started = time.perf_counter()
    stage_start("PIPELINE", "DETECTOR_TRAIN_OOF")
    train_oof_policy_rows = build_detector_train_oof_policy_rows(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_cv_config=config.detector_cv,
        detector_model_config=config.detector_model,
    )
    detector_oof_elapsed = time.perf_counter() - detector_oof_started
    stage_done("PIPELINE", "DETECTOR_TRAIN_OOF", elapsed_sec=detector_oof_elapsed)

    detector_val_started = time.perf_counter()
    stage_start("PIPELINE", "DETECTOR_VAL_POLICY")
    _, val_policy_rows = build_detector_val_policy_rows(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_model_config=config.detector_model,
    )
    selected_detector_policy, detector_policy_sweep_df = select_detector_policy(
        val_policy_rows,
        config.detector_policy,
        search_config=config.search_detector_policy,
        window_start=config.splits.train_end,
        window_end=config.splits.val_end,
    )
    train_oof_candidate_signals_df, train_oof_episode_policy_summary_df = (
        apply_episode_aware_detector_policy(
            train_oof_policy_rows, selected_detector_policy
        )
    )
    detector_train_oof_window_days = compute_eval_window_days_from_policy_rows(
        train_oof_policy_rows
    )
    detector_train_oof_policy_metrics = build_detector_policy_metrics(
        train_oof_candidate_signals_df,
        train_oof_episode_policy_summary_df,
        window_days=detector_train_oof_window_days,
    )
    val_candidate_signals_df, val_episode_policy_summary_df = (
        apply_episode_aware_detector_policy(val_policy_rows, selected_detector_policy)
    )
    detector_val_policy_metrics = build_detector_policy_metrics(
        val_candidate_signals_df,
        val_episode_policy_summary_df,
        window_start=config.splits.train_end,
        window_end=config.splits.val_end,
    )
    detector_val_target_metrics = build_detector_target_metrics(val_policy_rows)
    detector_val_elapsed = time.perf_counter() - detector_val_started
    stage_done("PIPELINE", "DETECTOR_VAL_POLICY", elapsed_sec=detector_val_elapsed)

    detector_test_started = time.perf_counter()
    stage_start("PIPELINE", "DETECTOR_TEST")
    detector_model_test, test_policy_rows = build_detector_test_policy_rows(
        dataset_df=detector_dataset,
        split_bounds=config.splits,
        resolver_config=config.resolver,
        event_opener_config=config.event_opener,
        detector_model_config=config.detector_model,
    )
    test_candidate_signals_df, test_episode_policy_summary_df = (
        apply_episode_aware_detector_policy(test_policy_rows, selected_detector_policy)
    )
    detector_test_policy_metrics = build_detector_policy_metrics(
        test_candidate_signals_df,
        test_episode_policy_summary_df,
        window_start=config.splits.val_end,
        window_end=config.splits.test_end,
    )
    detector_test_target_metrics = build_detector_target_metrics(test_policy_rows)
    detector_test_elapsed = time.perf_counter() - detector_test_started
    test_window_days = float(
        (pd.Timestamp(config.splits.test_end) - pd.Timestamp(config.splits.val_end))
        / pd.Timedelta(days=1)
    )
    test_signals_per_30d_estimate = (
        len(test_candidate_signals_df) * 30.0 / max(test_window_days, 1e-9)
    )
    log_info(
        "PIPELINE",
        (
            f"DETECTOR_TEST summary elapsed_sec_total={detector_test_elapsed:.3f} "
            f"policy_rows_total={len(test_policy_rows)} candidate_signals_total={len(test_candidate_signals_df)} "
            f"episodes_total={test_policy_rows['episode_id'].nunique() if not test_policy_rows.empty else 0} "
            f"signals_per_30d_estimate={test_signals_per_30d_estimate:.6f}"
        ),
    )
    stage_done("PIPELINE", "DETECTOR_TEST", elapsed_sec=detector_test_elapsed)

    gate_val_started = time.perf_counter()
    stage_start("PIPELINE", "GATE_VAL")
    val_intervals = _derive_1m_stage_intervals(
        [train_oof_candidate_signals_df, val_candidate_signals_df], config.execution
    )
    val_symbols, val_start, val_end = _summarize_1m_intervals(val_intervals)
    log_info(
        "PIPELINE",
        (
            f"GATE_VAL context candidate_signals_train_oof={len(train_oof_candidate_signals_df)} "
            f"candidate_signals_val={len(val_candidate_signals_df)} "
            f"derived_1m_symbols_total={len(val_symbols)} derived_1m_start={val_start} derived_1m_end={val_end}"
        ),
    )
    val_1m_load_started = time.perf_counter()
    raw_1m_val = market_loader.load_1m_ohlcv_intervals(val_intervals)
    bars_1m_val = prepare_intraday_bars_frame(raw_1m_val, "1m")
    execution_market_view_val = build_execution_market_view(bars_15m, bars_1m_val)
    val_1m_load_elapsed = time.perf_counter() - val_1m_load_started
    log_info(
        "PIPELINE",
        (
            f"GATE_VAL 1m load done rows_1m_total={len(bars_1m_val)} "
            f"symbols_total={bars_1m_val['symbol'].nunique() if not bars_1m_val.empty else 0} "
            f"elapsed_sec={val_1m_load_elapsed:.3f}"
        ),
    )
    (
        _,
        val_scored_signals_df,
        gate_threshold_sweep_diagnostic_df,
        gate_dataset_train_oof_df,
        gate_dataset_val_df,
        gate_status_val,
        selected_gate_model_config,
    ) = build_gate_val_scored_signals_and_datasets(
        train_oof_candidate_signals_df,
        val_candidate_signals_df,
        token_state_tradable,
        reference_state,
        breadth_state,
        config.gate_model,
        config.gate_config.block_threshold,
        bars_15m,
        bars_1m_val,
        config.execution,
        bars_1s_fetcher,
        execution_market_view=execution_market_view_val,
        search_model_config=config.search_gate_model,
        search_threshold_config=config.search_gate_threshold,
        window_start=config.splits.train_end,
        window_end=config.splits.val_end,
    )
    train_rows = int(
        (
            (gate_dataset_train_oof_df["score_source"] == "train_oof")
            & gate_dataset_train_oof_df["gate_trainable_signal"].astype(bool)
        ).sum()
    ) if not gate_dataset_train_oof_df.empty else 0
    val_rows = int(
        (gate_dataset_val_df["score_source"] == "val_forward").sum()
    ) if not gate_dataset_val_df.empty else 0
    positive_rate_train = (
        float(
            pd.to_numeric(
                gate_dataset_train_oof_df.loc[
                    gate_dataset_train_oof_df["gate_trainable_signal"].astype(bool),
                    "target_block_signal",
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .mean()
        )
        if (
            not gate_dataset_train_oof_df.empty
            and bool(gate_dataset_train_oof_df["gate_trainable_signal"].astype(bool).any())
        )
        else 0.0
    )
    if gate_status_val == "enabled":
        selected_gate_threshold, gate_threshold_sweep_execution_df = (
            select_gate_block_threshold_execution_aware(
                scored_signals_df=val_scored_signals_df,
                base_block_threshold=config.gate_config.block_threshold,
                bars_15m_df=bars_15m,
                bars_1m_df=bars_1m_val,
                execution_contract=config.execution,
                bars_1s_fetcher=bars_1s_fetcher,
                execution_market_view=execution_market_view_val,
                search_config=config.search_gate_threshold,
                window_start=config.splits.train_end,
                window_end=config.splits.val_end,
            )
        )
        if selected_gate_threshold is None:
            selected_gate_mode = "disabled_by_selection"
            val_gate_decisions_df = val_scored_signals_df.copy()
            val_gate_decisions_df["gate_decision"] = "keep"
            val_gate_decisions_df["gate_block_threshold"] = pd.NA
        else:
            selected_gate_mode = "threshold"
            val_gate_decisions_df, _ = apply_gate_block_threshold(
                val_scored_signals_df, selected_gate_threshold
            )
    else:
        selected_gate_threshold = float(config.gate_config.block_threshold)
        gate_threshold_sweep_execution_df = gate_threshold_sweep_diagnostic_df.copy()
        selected_gate_mode = "disabled_no_data"
        val_gate_decisions_df, _ = apply_gate_block_threshold(
            val_scored_signals_df, selected_gate_threshold
        )
    gate_val_elapsed = time.perf_counter() - gate_val_started
    selected_gate_threshold_log = (
        f"{float(selected_gate_threshold):.6f}"
        if selected_gate_threshold is not None
        else "None"
    )
    log_info(
        "PIPELINE",
        (
            f"GATE_VAL summary elapsed_sec_total={gate_val_elapsed:.3f} gate_status={gate_status_val} "
            f"selected_gate_mode={selected_gate_mode} selected_gate_threshold={selected_gate_threshold_log} "
            f"candidate_signals_val={len(val_scored_signals_df)} "
            f"candidate_signals_after_gate={int((val_gate_decisions_df['gate_decision'] == 'keep').sum())}"
        ),
    )
    stage_done("PIPELINE", "GATE_VAL", elapsed_sec=gate_val_elapsed)

    gate_test_started = time.perf_counter()
    stage_start("PIPELINE", "GATE_TEST")
    test_intervals = _derive_1m_stage_intervals(
        [train_oof_candidate_signals_df, val_candidate_signals_df, test_candidate_signals_df],
        config.execution,
    )
    test_symbols, test_start, test_end = _summarize_1m_intervals(test_intervals)
    history_rows_for_context = len(train_oof_candidate_signals_df) + len(val_candidate_signals_df)
    log_info(
        "PIPELINE",
        (
            f"GATE_TEST context candidate_signals_test={len(test_candidate_signals_df)} "
            f"history_rows_for_context={history_rows_for_context} "
            f"derived_1m_symbols_total={len(test_symbols)} derived_1m_start={test_start} derived_1m_end={test_end}"
        ),
    )
    test_1m_load_started = time.perf_counter()
    raw_1m_test = market_loader.load_1m_ohlcv_intervals(test_intervals)
    bars_1m_test = prepare_intraday_bars_frame(raw_1m_test, "1m")
    execution_market_view_test = build_execution_market_view(bars_15m, bars_1m_test)
    test_1m_load_elapsed = time.perf_counter() - test_1m_load_started
    log_info(
        "PIPELINE",
        (
            f"GATE_TEST 1m load done rows_1m_total={len(bars_1m_test)} "
            f"symbols_total={bars_1m_test['symbol'].nunique() if not bars_1m_test.empty else 0} "
            f"elapsed_sec={test_1m_load_elapsed:.3f}"
        ),
    )
    gate_model_test, test_scored_signals_df, gate_dataset_test_df, gate_status_test = (
        build_gate_test_scored_signals(
            train_oof_candidate_signals_df,
            test_candidate_signals_df,
            val_candidate_signals_df,
            token_state_tradable,
            reference_state,
            breadth_state,
            selected_gate_model_config,
            bars_15m,
            bars_1m_test,
            config.execution,
            bars_1s_fetcher,
            force_disabled_no_data=(gate_status_val != "enabled"),
            execution_market_view=execution_market_view_test,
        )
    )
    if selected_gate_threshold is None:
        test_gate_decisions_df = test_scored_signals_df.copy()
        test_gate_decisions_df["gate_decision"] = "keep"
        test_gate_decisions_df["gate_block_threshold"] = pd.NA
    else:
        test_gate_decisions_df, _ = apply_gate_block_threshold(
            test_scored_signals_df, selected_gate_threshold
        )
    test_rows = int(
        (gate_dataset_test_df["score_source"] == "test_forward").sum()
    ) if not gate_dataset_test_df.empty else 0
    train_rows_for_gate_test = int(
        (
            (gate_dataset_test_df["score_source"] == "train_oof")
            & gate_dataset_test_df["gate_trainable_signal"].astype(bool)
        ).sum()
    ) if not gate_dataset_test_df.empty and "score_source" in gate_dataset_test_df.columns else 0
    kept_total = int((test_gate_decisions_df["gate_decision"] == "keep").sum())
    blocked_total = int((test_gate_decisions_df["gate_decision"] == "block").sum())
    keep_rate = kept_total / len(test_gate_decisions_df) if len(test_gate_decisions_df) > 0 else 0.0
    block_rate = blocked_total / len(test_gate_decisions_df) if len(test_gate_decisions_df) > 0 else 0.0
    log_info(
        "PIPELINE",
        (
            f"GATE_TEST threshold apply kept_total={kept_total} blocked_total={blocked_total} "
            f"keep_rate={keep_rate:.6f} block_rate={block_rate:.6f}"
        ),
    )
    gate_test_elapsed = time.perf_counter() - gate_test_started
    stage_done("PIPELINE", "GATE_TEST", elapsed_sec=gate_test_elapsed)

    execution_replay_started = time.perf_counter()
    stage_start("PIPELINE", "EXECUTION_REPLAY")
    val_replay_started = time.perf_counter()
    counterfactual_cols = [
        "counterfactual_execution_status",
        "counterfactual_trade_outcome",
        "counterfactual_trade_pnl_pct",
        "counterfactual_exit_time",
    ]
    val_replay_input_df = val_gate_decisions_df.drop(
        columns=[c for c in counterfactual_cols if c in val_gate_decisions_df.columns]
    )
    log_info(
        "PIPELINE",
        f"split=val execution replay start decisions_total={len(val_replay_input_df)} symbols_total={val_replay_input_df['symbol'].nunique() if not val_replay_input_df.empty else 0}",
    )
    val_execution_decisions_df, val_executed_signals_df = (
        replay_short_signals_with_symbol_lock(
            val_replay_input_df,
            bars_15m,
            bars_1m_val,
            config.execution,
            bars_1s_fetcher,
            emit_summary_log=False,
            market_view=execution_market_view_val,
        )
    )
    val_replay_elapsed = time.perf_counter() - val_replay_started
    val_rate = (
        len(val_replay_input_df) / val_replay_elapsed if val_replay_elapsed > 0 else 0.0
    )
    log_info(
        "PIPELINE",
        (
            f"split=val execution replay done executed_total={len(val_executed_signals_df)} "
            f"blocked_gate={int((val_execution_decisions_df['execution_status'] == 'blocked_gate').sum())} "
            f"blocked_symbol_lock={int((val_execution_decisions_df['execution_status'] == 'blocked_symbol_lock').sum())} "
            f"ambiguous={int((val_execution_decisions_df['trade_outcome'] == 'ambiguous').sum())} "
            f"elapsed_sec={val_replay_elapsed:.3f} decisions_per_sec={val_rate:.3f}"
        ),
    )
    test_replay_started = time.perf_counter()
    test_replay_input_df = test_gate_decisions_df.drop(
        columns=[c for c in counterfactual_cols if c in test_gate_decisions_df.columns]
    )
    log_info(
        "PIPELINE",
        f"split=test execution replay start decisions_total={len(test_replay_input_df)} symbols_total={test_replay_input_df['symbol'].nunique() if not test_replay_input_df.empty else 0}",
    )
    test_execution_decisions_df, test_executed_signals_df = (
        replay_short_signals_with_symbol_lock(
            test_replay_input_df,
            bars_15m,
            bars_1m_test,
            config.execution,
            bars_1s_fetcher,
            emit_summary_log=False,
            market_view=execution_market_view_test,
        )
    )
    test_replay_elapsed = time.perf_counter() - test_replay_started
    test_rate = (
        len(test_replay_input_df) / test_replay_elapsed if test_replay_elapsed > 0 else 0.0
    )
    log_info(
        "PIPELINE",
        (
            f"split=test execution replay done executed_total={len(test_executed_signals_df)} "
            f"blocked_gate={int((test_execution_decisions_df['execution_status'] == 'blocked_gate').sum())} "
            f"blocked_symbol_lock={int((test_execution_decisions_df['execution_status'] == 'blocked_symbol_lock').sum())} "
            f"ambiguous={int((test_execution_decisions_df['trade_outcome'] == 'ambiguous').sum())} "
            f"elapsed_sec={test_replay_elapsed:.3f} decisions_per_sec={test_rate:.3f}"
        ),
    )
    val_counterfactual_df = val_gate_decisions_df.loc[
        :,
        [
            "signal_id",
            "counterfactual_execution_status",
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
            "counterfactual_exit_time",
        ],
    ].copy()
    test_counterfactual_df = test_gate_decisions_df.loc[
        :,
        [
            "signal_id",
            "counterfactual_execution_status",
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
            "counterfactual_exit_time",
        ],
    ].copy()
    if {
        "counterfactual_trade_outcome",
        "counterfactual_trade_pnl_pct",
    }.issubset(val_execution_decisions_df.columns):
        val_execution_decisions_enriched_df = val_execution_decisions_df.copy()
    else:
        val_execution_decisions_enriched_df = val_execution_decisions_df.merge(
            val_counterfactual_df,
            on="signal_id",
            how="left",
            validate="one_to_one",
        )
    if {
        "counterfactual_trade_outcome",
        "counterfactual_trade_pnl_pct",
    }.issubset(test_execution_decisions_df.columns):
        test_execution_decisions_enriched_df = test_execution_decisions_df.copy()
    else:
        test_execution_decisions_enriched_df = test_execution_decisions_df.merge(
            test_counterfactual_df,
            on="signal_id",
            how="left",
            validate="one_to_one",
        )
    val_decision_summary = build_gate_execution_decision_summary(
        val_execution_decisions_enriched_df
    )
    test_decision_summary = build_gate_execution_decision_summary(
        test_execution_decisions_enriched_df
    )
    execution_replay_elapsed = time.perf_counter() - execution_replay_started
    stage_done("PIPELINE", "EXECUTION_REPLAY", elapsed_sec=execution_replay_elapsed)

    execution_reports_started = time.perf_counter()
    stage_start("PIPELINE", "EXECUTION_REPORTS")
    val_window_6h = build_execution_window_report(val_executed_signals_df, 6)
    val_window_24h = build_execution_window_report(val_executed_signals_df, 24)
    val_metrics = build_execution_metrics(
        val_executed_signals_df,
        window_start=config.splits.train_end,
        window_end=config.splits.val_end,
        precomputed_window_reports={6: val_window_6h, 24: val_window_24h},
    )
    val_symbol_report = build_execution_symbol_report(val_executed_signals_df)
    val_monthly_report = build_execution_monthly_report(val_executed_signals_df)
    test_window_6h = build_execution_window_report(test_executed_signals_df, 6)
    test_window_24h = build_execution_window_report(test_executed_signals_df, 24)
    test_metrics = build_execution_metrics(
        test_executed_signals_df,
        window_start=config.splits.val_end,
        window_end=config.splits.test_end,
        precomputed_window_reports={6: test_window_6h, 24: test_window_24h},
    )
    test_symbol_report = build_execution_symbol_report(test_executed_signals_df)
    test_monthly_report = build_execution_monthly_report(test_executed_signals_df)
    gate_deciles_val = build_gate_decile_report(val_scored_signals_df)
    gate_deciles_test = build_gate_decile_report(test_scored_signals_df)
    val_candidate_signal_strength = build_candidate_signal_strength_report(
        val_scored_signals_df, config.execution
    )
    test_candidate_signal_strength = build_candidate_signal_strength_report(
        test_scored_signals_df, config.execution
    )
    val_gate_rank_quality = build_gate_rank_quality_report(val_scored_signals_df)
    test_gate_rank_quality = build_gate_rank_quality_report(test_scored_signals_df)
    log_info(
        "PIPELINE",
        (
            f"split=val reports summary signals={int(val_metrics.get('signals', 0.0))} "
            f"signals_per_30d={float(val_metrics.get('signals_per_30d', 0.0)):.6f} "
            f"pnl_sum={float(val_metrics.get('pnl_sum', 0.0)):.6f} "
            f"worst_6h={float(val_metrics.get('worst_6h_pnl', 0.0)):.6f} "
            f"worst_24h={float(val_metrics.get('worst_24h_pnl', 0.0)):.6f} "
            f"max_losing_streak={float(val_metrics.get('max_losing_streak', 0.0)):.0f}"
        ),
    )
    log_info(
        "PIPELINE",
        (
            f"split=test reports summary signals={int(test_metrics.get('signals', 0.0))} "
            f"signals_per_30d={float(test_metrics.get('signals_per_30d', 0.0)):.6f} "
            f"pnl_sum={float(test_metrics.get('pnl_sum', 0.0)):.6f} "
            f"worst_6h={float(test_metrics.get('worst_6h_pnl', 0.0)):.6f} "
            f"worst_24h={float(test_metrics.get('worst_24h_pnl', 0.0)):.6f} "
            f"max_losing_streak={float(test_metrics.get('max_losing_streak', 0.0)):.0f}"
        ),
    )
    execution_reports_elapsed = time.perf_counter() - execution_reports_started
    stage_done("PIPELINE", "EXECUTION_REPORTS", elapsed_sec=execution_reports_elapsed)

    artifacts_started = time.perf_counter()
    stage_start("PIPELINE", "ARTIFACTS_SAVE")
    log_info("ARTIFACTS", "artifacts save started")
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
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={prepared_dir / 'event_quality_report.json'}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={prepared_dir / 'episodes.parquet'} rows={len(episodes)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={prepared_dir / 'decision_rows.parquet'} rows={len(decision_rows)}",
    )

    _save_df_and_log(detector_dataset, detector_dir / "dataset.parquet")
    _save_json_and_log(
        build_detector_feature_manifest(), detector_dir / "feature_manifest.json"
    )
    _save_df_and_log(detector_policy_sweep_df, detector_dir / "policy_sweep_val.csv")
    _save_json_and_log(
        _policy_to_dict(selected_detector_policy), detector_dir / "selected_policy.json"
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={detector_dir / 'dataset.parquet'} rows={len(detector_dataset)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={detector_dir / 'policy_sweep_val.csv'} rows={len(detector_policy_sweep_df)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={detector_dir / 'selected_policy.json'}",
    )
    _save_df_and_log(
        train_oof_policy_rows, detector_dir / "train_oof_policy_rows.parquet"
    )
    _save_df_and_log(
        train_oof_candidate_signals_df,
        detector_dir / "train_oof_candidate_signals.parquet",
    )
    _save_df_and_log(
        train_oof_episode_policy_summary_df,
        detector_dir / "train_oof_episode_policy_summary.parquet",
    )
    _save_json_and_log(
        detector_train_oof_policy_metrics,
        detector_dir / "train_oof_policy_metrics.json",
    )
    _save_df_and_log(val_policy_rows, detector_dir / "val_policy_rows.parquet")
    _save_df_and_log(
        val_candidate_signals_df, detector_dir / "val_candidate_signals.parquet"
    )
    _save_df_and_log(
        val_episode_policy_summary_df,
        detector_dir / "val_episode_policy_summary.parquet",
    )
    _save_json_and_log(
        detector_val_policy_metrics, detector_dir / "val_policy_metrics.json"
    )
    _save_json_and_log(
        detector_val_target_metrics, detector_dir / "val_target_metrics.json"
    )
    _save_df_and_log(test_policy_rows, detector_dir / "test_policy_rows.parquet")
    _save_df_and_log(
        test_candidate_signals_df, detector_dir / "test_candidate_signals.parquet"
    )
    _save_df_and_log(
        test_episode_policy_summary_df,
        detector_dir / "test_episode_policy_summary.parquet",
    )
    _save_json_and_log(
        detector_test_policy_metrics, detector_dir / "test_policy_metrics.json"
    )
    _save_json_and_log(
        detector_test_target_metrics, detector_dir / "test_target_metrics.json"
    )
    _save_model_and_log(detector_model_test, detector_dir / "model_train_only.cbm")

    _save_df_and_log(val_scored_signals_df, gate_dir / "val_scored_signals.parquet")
    _save_json_and_log(
        build_gate_feature_manifest(), gate_dir / "feature_manifest.json"
    )
    _save_df_and_log(gate_dataset_train_oof_df, gate_dir / "dataset_train_oof.parquet")
    _save_df_and_log(gate_dataset_val_df, gate_dir / "dataset_val.parquet")
    _save_df_and_log(gate_dataset_test_df, gate_dir / "dataset_test.parquet")
    _save_df_and_log(
        gate_threshold_sweep_diagnostic_df,
        gate_dir / "threshold_sweep_val_diagnostic.csv",
    )
    _save_df_and_log(
        gate_threshold_sweep_execution_df,
        gate_dir / "threshold_sweep_val_execution.csv",
    )
    _save_json_and_log(selected_gate_threshold, gate_dir / "selected_threshold.json")
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={gate_dir / 'dataset_train_oof.parquet'} rows={len(gate_dataset_train_oof_df)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={gate_dir / 'threshold_sweep_val_execution.csv'} rows={len(gate_threshold_sweep_execution_df)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={gate_dir / 'selected_threshold.json'}",
    )
    _save_df_and_log(gate_deciles_val, gate_dir / "gate_deciles.csv")
    _save_df_and_log(gate_deciles_test, gate_dir / "gate_deciles_test.csv")
    _save_df_and_log(test_scored_signals_df, gate_dir / "test_scored_signals.parquet")
    if gate_model_test is not None:
        _save_model_and_log(gate_model_test, gate_dir / "model_train_oof.cbm")

    _save_df_and_log(val_gate_decisions_df, eval_val_dir / "candidate_signals.parquet")
    _save_df_and_log(
        val_execution_decisions_df, eval_val_dir / "execution_decisions.parquet"
    )
    _save_df_and_log(val_executed_signals_df, eval_val_dir / "executed_signals.csv")
    _save_json_and_log(val_metrics, eval_val_dir / "metrics.json")
    _save_json_and_log(val_decision_summary, eval_val_dir / "decision_summary.json")
    _save_df_and_log(val_window_6h, eval_val_dir / "window_report_6h.csv")
    _save_df_and_log(val_window_24h, eval_val_dir / "window_report_24h.csv")
    _save_df_and_log(val_symbol_report, eval_val_dir / "symbol_report.csv")
    _save_df_and_log(val_monthly_report, eval_val_dir / "monthly_report.csv")

    _save_df_and_log(
        test_gate_decisions_df, eval_test_dir / "candidate_signals.parquet"
    )
    holdout_signals_path = _save_df_and_log(
        test_executed_signals_df, eval_test_dir / "test_signals_holdout.csv"
    )
    _save_df_and_log(
        test_execution_decisions_df, eval_test_dir / "execution_decisions.parquet"
    )
    _save_json_and_log(test_metrics, eval_test_dir / "metrics_holdout.json")
    _save_json_and_log(test_decision_summary, eval_test_dir / "decision_summary.json")
    _save_df_and_log(test_window_6h, eval_test_dir / "window_report_6h.csv")
    _save_df_and_log(test_window_24h, eval_test_dir / "window_report_24h.csv")
    _save_df_and_log(test_symbol_report, eval_test_dir / "symbol_report.csv")
    _save_df_and_log(test_monthly_report, eval_test_dir / "monthly_report.csv")
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={eval_test_dir / 'test_signals_holdout.csv'} rows={len(test_executed_signals_df)}",
    )
    log_info(
        "ARTIFACTS",
        f"key artifact saved path={eval_test_dir / 'metrics_holdout.json'}",
    )

    run_summary = {
        "run_id": run_context.run_id,
        "run_dir": str(run_context.run_dir),
        "config_path": str(Path(config_path).resolve()),
        "selected_detector_policy": _policy_to_dict(selected_detector_policy),
        "selected_gate_model": _gate_model_to_dict(selected_gate_model_config),
        "selected_gate_mode": selected_gate_mode,
        "selected_gate_threshold": (
            float(selected_gate_threshold)
            if selected_gate_threshold is not None
            else None
        ),
        "gate_status": (
            "disabled_no_data"
            if (gate_status_val != "enabled" or gate_status_test != "enabled")
            else "enabled"
        ),
        "gate_status_val": gate_status_val,
        "gate_status_test": gate_status_test,
        "event_quality_report": event_quality_report,
        "detector_val_policy_metrics": detector_val_policy_metrics,
        "detector_train_oof_policy_metrics": detector_train_oof_policy_metrics,
        "detector_test_policy_metrics": detector_test_policy_metrics,
        "val_candidate_signal_strength": val_candidate_signal_strength,
        "test_candidate_signal_strength": test_candidate_signal_strength,
        "val_gate_rank_quality": val_gate_rank_quality,
        "test_gate_rank_quality": test_gate_rank_quality,
        "val_decision_summary": val_decision_summary,
        "test_decision_summary": test_decision_summary,
        "val_execution_metrics": val_metrics,
        "test_execution_metrics": test_metrics,
    }
    summary_path = _save_json_and_log(run_summary, reports_dir / "run_summary.json")
    log_info("ARTIFACTS", f"key artifact saved path={reports_dir / 'run_summary.json'}")
    artifacts_elapsed = time.perf_counter() - artifacts_started
    log_info(
        "ARTIFACTS",
        f"artifacts save completed files_key_total=12 elapsed_sec={artifacts_elapsed:.3f}",
    )
    stage_done("PIPELINE", "ARTIFACTS_SAVE", elapsed_sec=artifacts_elapsed)
    gate_status = (
        "disabled_no_data"
        if (gate_status_val != "enabled" or gate_status_test != "enabled")
        else "enabled"
    )
    total_elapsed = time.perf_counter() - started
    log_info(
        "PIPELINE",
        (
            f"FULL_RUN summary elapsed_total_sec={total_elapsed:.3f} gate_status={gate_status} "
            f"test_candidate_signals={len(test_gate_decisions_df)} test_executed_signals={len(test_executed_signals_df)} "
            f"signals_per_30d={float(test_metrics.get('signals_per_30d', 0.0)):.6f} "
            f"pnl_sum={float(test_metrics.get('pnl_sum', 0.0)):.6f} "
            f"worst_24h={float(test_metrics.get('worst_24h_pnl', 0.0)):.6f} "
            f"holdout_path={holdout_signals_path} summary_path={summary_path}"
        ),
    )
    stage_done("PIPELINE", "FULL_RUN", elapsed_sec=total_elapsed)
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
    return saved_path


def _save_json_and_log(payload: Any, path: Path) -> Path:
    saved_path = save_json_artifact(payload, path)
    return saved_path


def _save_model_and_log(model: Any, path: Path) -> Path:
    if not hasattr(model, "save_model"):
        raise TypeError(f"model object does not support save_model: {type(model)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def _policy_to_dict(policy: Any) -> dict[str, float]:
    return {
        "arm_score_min": float(policy.arm_score_min),
        "fire_score_floor": float(policy.fire_score_floor),
        "turn_down_delta": float(policy.turn_down_delta),
    }


def _gate_model_to_dict(model_config: Any) -> dict[str, float]:
    return {
        "iterations": float(model_config.iterations),
        "depth": float(model_config.depth),
        "learning_rate": float(model_config.learning_rate),
        "l2_leaf_reg": float(model_config.l2_leaf_reg),
        "random_seed": float(model_config.random_seed),
        "tp_row_weight": float(model_config.tp_row_weight),
        "sl_row_weight": float(model_config.sl_row_weight),
    }


def _derive_1m_stage_intervals(
    candidate_frames: list, execution_contract
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    horizon = pd.Timedelta(minutes=int(execution_contract.max_hold_bars) * 15 + 15)
    for frame in candidate_frames:
        if frame is None or frame.empty:
            continue
        required = {"symbol", "entry_bar_open_time"}
        if not required.issubset(frame.columns):
            continue
        local = frame.loc[:, ["symbol", "entry_bar_open_time"]].copy()
        local["entry_bar_open_time"] = pd.to_datetime(
            local["entry_bar_open_time"], utc=True, errors="coerce"
        )
        local = local.dropna(subset=["entry_bar_open_time"])
        if local.empty:
            continue
        local["symbol"] = (
            local["symbol"].astype(str).str.strip().str.upper().replace("", pd.NA)
        )
        local = local.dropna(subset=["symbol"]).reset_index(drop=True)
        for row in local.itertuples(index=False):
            symbol = str(row.symbol)
            start = pd.Timestamp(row.entry_bar_open_time)
            end = start + horizon
            by_symbol.setdefault(symbol, []).append((start, end))
    merged: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for symbol, intervals in by_symbol.items():
        if not intervals:
            continue
        ordered = sorted(intervals, key=lambda item: item[0])
        merged_symbol: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cur_start, cur_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                continue
            merged_symbol.append((cur_start, cur_end))
            cur_start, cur_end = start, end
        merged_symbol.append((cur_start, cur_end))
        merged[symbol] = merged_symbol
    return dict(sorted(merged.items()))


def _summarize_1m_intervals(
    symbol_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    if not symbol_intervals:
        now_utc = pd.Timestamp.utcnow()
        return tuple(), now_utc, now_utc + pd.Timedelta(minutes=1)
    symbols = tuple(sorted(symbol_intervals.keys()))
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for intervals in symbol_intervals.values():
        for start, end in intervals:
            starts.append(pd.Timestamp(start))
            ends.append(pd.Timestamp(end))
    return symbols, min(starts), max(ends)
