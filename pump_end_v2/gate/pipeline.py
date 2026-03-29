import pandas as pd
from catboost import CatBoostClassifier, CatBoostError

from pump_end_v2.config import GateModelConfig, GateThresholdSearchConfig
from pump_end_v2.contracts import ExecutionContract
from pump_end_v2.gate.dataset import GATE_TARGET_META_COLUMNS, build_gate_dataset
from pump_end_v2.gate.feature_view import (
    GATE_FEATURE_COLUMNS,
    GATE_IDENTITY_COLUMNS,
    build_gate_feature_view,
)
from pump_end_v2.gate.model import build_gate_model, fit_gate_model, predict_gate_scores
from pump_end_v2.gate.threshold import (
    attach_counterfactual_execution_outcomes,
    sweep_gate_block_threshold,
)
from pump_end_v2.logging import log_info

_SCORED_OUTPUT_COLUMNS: tuple[str, ...] = (
    *GATE_IDENTITY_COLUMNS,
    "p_block",
    "target_block_signal",
    "block_reason",
    "signal_quality_h32",
    "gate_trainable_signal",
    "target_good_short_now",
    "target_reason",
    "future_outcome_class",
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
_GATE_STATUS_ENABLED = "enabled"
_GATE_STATUS_DISABLED_NO_DATA = "disabled_no_data"


def _is_informative_gate_feature(series: pd.Series) -> bool:
    non_na = series.dropna()
    if non_na.empty:
        return False
    return bool(non_na.nunique(dropna=True) > 1)


def _is_gate_trainable(train_fit_df: pd.DataFrame) -> bool:
    if train_fit_df.empty:
        return False
    if "target_block_signal" not in train_fit_df.columns:
        return False
    target_unique = (
        pd.to_numeric(train_fit_df["target_block_signal"], errors="coerce")
        .dropna()
        .nunique()
    )
    if int(target_unique) < 2:
        return False
    informative_features = 0
    for column in GATE_FEATURE_COLUMNS:
        if column not in train_fit_df.columns:
            continue
        if _is_informative_gate_feature(train_fit_df[column]):
            informative_features += 1
    return informative_features > 0


def build_gate_val_scored_signals_and_datasets(
    train_oof_candidate_signals_df: pd.DataFrame,
    val_candidate_signals_df: pd.DataFrame,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    gate_model_config: GateModelConfig,
    base_block_threshold: float,
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: object | None = None,
    execution_market_view: object | None = None,
    search_threshold_config: GateThresholdSearchConfig | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_days: float | None = None,
) -> tuple[
    CatBoostClassifier | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    train_oof_with_execution_df = _enrich_with_counterfactual(
        train_oof_candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label="train_oof",
        execution_market_view=execution_market_view,
    )
    val_with_execution_df = _enrich_with_counterfactual(
        val_candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label="val",
        execution_market_view=execution_market_view,
    )
    val_history_df = _slice_history_window(
        train_oof_with_execution_df, val_candidate_signals_df, hours=24
    )
    train_feature_view_df = build_gate_feature_view(
        candidate_signals_df=train_oof_with_execution_df,
        history_candidate_signals_df=None,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
    )
    val_feature_view_df = build_gate_feature_view(
        candidate_signals_df=val_with_execution_df,
        history_candidate_signals_df=val_history_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
    )
    train_gate_dataset_df = build_gate_dataset(
        train_feature_view_df, train_oof_with_execution_df
    )
    val_gate_dataset_df = build_gate_dataset(val_feature_view_df, val_with_execution_df)
    train_fit_df = train_gate_dataset_df[
        (train_gate_dataset_df["score_source"] == "train_oof")
        & train_gate_dataset_df["gate_trainable_signal"].astype(bool)
    ].copy()
    val_score_df = val_gate_dataset_df[
        val_gate_dataset_df["score_source"] == "val_forward"
    ].copy()
    if (not _is_gate_trainable(train_fit_df)) or val_score_df.empty:
        val_scored_signals_df = _build_disabled_scored_signals(
            val_with_execution_df, "val_forward"
        )
        val_scored_signals_df = _attach_counterfactual_columns(
            val_scored_signals_df, val_with_execution_df
        )
        threshold_sweep_df = sweep_gate_block_threshold(
            scored_signals_df=val_scored_signals_df,
            base_block_threshold=base_block_threshold,
            search_config=search_threshold_config,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        log_info(
            "GATE",
            (
                "gate val wrapper fallback "
                f"status={_GATE_STATUS_DISABLED_NO_DATA} train_rows={len(train_fit_df)} val_rows={len(val_score_df)} "
                f"scored_rows={len(val_scored_signals_df)}"
            ),
        )
        return (
            None,
            val_scored_signals_df,
            threshold_sweep_df,
            train_gate_dataset_df,
            val_gate_dataset_df,
            _GATE_STATUS_DISABLED_NO_DATA,
        )
    model = build_gate_model(gate_model_config)
    try:
        fit_gate_model(
            model,
            train_fit_df,
            GATE_FEATURE_COLUMNS,
            "target_block_signal",
            tp_row_weight=2.0,
            sl_row_weight=1.0,
        )
    except CatBoostError:
        val_scored_signals_df = _build_disabled_scored_signals(
            val_with_execution_df, "val_forward"
        )
        val_scored_signals_df = _attach_counterfactual_columns(
            val_scored_signals_df, val_with_execution_df
        )
        threshold_sweep_df = sweep_gate_block_threshold(
            scored_signals_df=val_scored_signals_df,
            base_block_threshold=base_block_threshold,
            search_config=search_threshold_config,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
        )
        log_info(
            "GATE",
            (
                "gate val wrapper fallback "
                f"status={_GATE_STATUS_DISABLED_NO_DATA} train_rows={len(train_fit_df)} val_rows={len(val_score_df)} "
                f"scored_rows={len(val_scored_signals_df)}"
            ),
        )
        return (
            None,
            val_scored_signals_df,
            threshold_sweep_df,
            train_gate_dataset_df,
            val_gate_dataset_df,
            _GATE_STATUS_DISABLED_NO_DATA,
        )
    val_scored_signals_df = _score_gate_dataset_rows(model, val_score_df)
    val_scored_signals_df = _attach_counterfactual_columns(
        val_scored_signals_df, val_with_execution_df
    )
    threshold_sweep_df = sweep_gate_block_threshold(
        scored_signals_df=val_scored_signals_df,
        base_block_threshold=base_block_threshold,
        search_config=search_threshold_config,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
    )
    log_info(
        "GATE",
        (
            "gate val wrapper done "
            f"train_rows={len(train_fit_df)} val_rows={len(val_score_df)} scored_rows={len(val_scored_signals_df)}"
        ),
    )
    return (
        model,
        val_scored_signals_df,
        threshold_sweep_df,
        train_gate_dataset_df,
        val_gate_dataset_df,
        _GATE_STATUS_ENABLED,
    )


def build_gate_test_scored_signals(
    train_oof_candidate_signals_df: pd.DataFrame,
    test_candidate_signals_df: pd.DataFrame,
    history_candidate_signals_df: pd.DataFrame | None,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    gate_model_config: GateModelConfig,
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: object | None = None,
    force_disabled_no_data: bool = False,
    execution_market_view: object | None = None,
) -> tuple[CatBoostClassifier | None, pd.DataFrame, pd.DataFrame, str]:
    if force_disabled_no_data:
        test_with_execution_df = _enrich_with_counterfactual(
            test_candidate_signals_df,
            bars_15m_df,
            bars_1m_df,
            execution_contract,
            bars_1s_fetcher,
            split_label="test",
            execution_market_view=execution_market_view,
        )
        test_scored_signals_df = _build_disabled_scored_signals(
            test_with_execution_df, "test_forward"
        )
        test_scored_signals_df = _attach_counterfactual_columns(
            test_scored_signals_df, test_with_execution_df
        )
        log_info(
            "GATE",
            (
                "gate test wrapper fallback "
                f"status={_GATE_STATUS_DISABLED_NO_DATA} forced=1 scored_rows={len(test_scored_signals_df)}"
            ),
        )
        return (
            None,
            test_scored_signals_df,
            pd.DataFrame(),
            _GATE_STATUS_DISABLED_NO_DATA,
        )
    train_oof_with_execution_df = _enrich_with_counterfactual(
        train_oof_candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label="train_oof",
        execution_market_view=execution_market_view,
    )
    test_with_execution_df = _enrich_with_counterfactual(
        test_candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label="test",
        execution_market_view=execution_market_view,
    )
    history_with_execution_df = _enrich_with_counterfactual(
        history_candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label="test_history",
        execution_market_view=execution_market_view,
    )
    test_history_df = _slice_history_window(
        history_with_execution_df, test_candidate_signals_df, hours=24
    )
    train_feature_view_df = build_gate_feature_view(
        candidate_signals_df=train_oof_with_execution_df,
        history_candidate_signals_df=None,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
    )
    test_feature_view_df = build_gate_feature_view(
        candidate_signals_df=test_with_execution_df,
        history_candidate_signals_df=test_history_df,
        token_state_df=token_state_df,
        reference_state_df=reference_state_df,
        breadth_state_df=breadth_state_df,
    )
    train_gate_dataset_df = build_gate_dataset(
        train_feature_view_df, train_oof_with_execution_df
    )
    test_gate_dataset_df = build_gate_dataset(
        test_feature_view_df, test_with_execution_df
    )
    train_fit_df = train_gate_dataset_df[
        (train_gate_dataset_df["score_source"] == "train_oof")
        & train_gate_dataset_df["gate_trainable_signal"].astype(bool)
    ].copy()
    test_score_df = test_gate_dataset_df[
        test_gate_dataset_df["score_source"] == "test_forward"
    ].copy()
    if (not _is_gate_trainable(train_fit_df)) or test_score_df.empty:
        test_scored_signals_df = _build_disabled_scored_signals(
            test_with_execution_df, "test_forward"
        )
        test_scored_signals_df = _attach_counterfactual_columns(
            test_scored_signals_df, test_with_execution_df
        )
        log_info(
            "GATE",
            (
                "gate test wrapper fallback "
                f"status={_GATE_STATUS_DISABLED_NO_DATA} train_rows={len(train_fit_df)} test_rows={len(test_score_df)} "
                f"scored_rows={len(test_scored_signals_df)}"
            ),
        )
        return (
            None,
            test_scored_signals_df,
            test_gate_dataset_df,
            _GATE_STATUS_DISABLED_NO_DATA,
        )
    model = build_gate_model(gate_model_config)
    try:
        fit_gate_model(
            model,
            train_fit_df,
            GATE_FEATURE_COLUMNS,
            "target_block_signal",
            tp_row_weight=2.0,
            sl_row_weight=1.0,
        )
    except CatBoostError:
        test_scored_signals_df = _build_disabled_scored_signals(
            test_with_execution_df, "test_forward"
        )
        test_scored_signals_df = _attach_counterfactual_columns(
            test_scored_signals_df, test_with_execution_df
        )
        log_info(
            "GATE",
            (
                "gate test wrapper fallback "
                f"status={_GATE_STATUS_DISABLED_NO_DATA} train_rows={len(train_fit_df)} test_rows={len(test_score_df)} "
                f"scored_rows={len(test_scored_signals_df)}"
            ),
        )
        return (
            None,
            test_scored_signals_df,
            test_gate_dataset_df,
            _GATE_STATUS_DISABLED_NO_DATA,
        )
    test_scored_signals_df = _score_gate_dataset_rows(model, test_score_df)
    test_scored_signals_df = _attach_counterfactual_columns(
        test_scored_signals_df, test_with_execution_df
    )
    log_info(
        "GATE",
        (
            "gate test wrapper done "
            f"train_rows={len(train_fit_df)} test_rows={len(test_score_df)} scored_rows={len(test_scored_signals_df)}"
        ),
    )
    return model, test_scored_signals_df, test_gate_dataset_df, _GATE_STATUS_ENABLED


def _score_gate_dataset_rows(
    model: CatBoostClassifier, rows_df: pd.DataFrame
) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=list(_SCORED_OUTPUT_COLUMNS))
    score_df = predict_gate_scores(model, rows_df, GATE_FEATURE_COLUMNS)
    merged = rows_df.merge(
        score_df[["signal_id", "p_block"]],
        on="signal_id",
        how="left",
        validate="one_to_one",
    )
    required_cols = [
        *GATE_IDENTITY_COLUMNS,
        *GATE_TARGET_META_COLUMNS,
        "target_block_signal",
        "gate_trainable_signal",
    ]
    _require_columns(merged, required_cols, "scored_rows")
    return merged.loc[:, list(_SCORED_OUTPUT_COLUMNS)].reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _slice_history_window(
    history_candidate_signals_df: pd.DataFrame | None,
    current_candidate_signals_df: pd.DataFrame,
    hours: int,
) -> pd.DataFrame | None:
    if (
        history_candidate_signals_df is None
        or history_candidate_signals_df.empty
        or current_candidate_signals_df.empty
    ):
        return None
    history = history_candidate_signals_df.copy()
    current = current_candidate_signals_df.copy()
    history["context_bar_open_time"] = pd.to_datetime(
        history["context_bar_open_time"], utc=True, errors="raise"
    )
    current["context_bar_open_time"] = pd.to_datetime(
        current["context_bar_open_time"], utc=True, errors="raise"
    )
    current_start = pd.Timestamp(current["context_bar_open_time"].min())
    window_start = current_start - pd.Timedelta(hours=int(hours))
    out = history[
        (history["context_bar_open_time"] >= window_start)
        & (history["context_bar_open_time"] < current_start)
    ].copy()
    if out.empty:
        return None
    return out.reset_index(drop=True)


def _enrich_with_counterfactual(
    candidate_signals_df: pd.DataFrame | None,
    bars_15m_df: pd.DataFrame,
    bars_1m_df: pd.DataFrame,
    execution_contract: ExecutionContract,
    bars_1s_fetcher: object | None,
    split_label: str,
    execution_market_view: object | None = None,
) -> pd.DataFrame | None:
    if candidate_signals_df is None:
        return None
    if candidate_signals_df.empty:
        out = candidate_signals_df.copy()
        out["counterfactual_execution_status"] = pd.NA
        out["counterfactual_trade_outcome"] = pd.NA
        out["counterfactual_trade_pnl_pct"] = pd.NA
        out["counterfactual_exit_time"] = pd.NaT
        return out
    counterfactual_df = attach_counterfactual_execution_outcomes(
        candidate_signals_df,
        bars_15m_df,
        bars_1m_df,
        execution_contract,
        bars_1s_fetcher,
        split_label=split_label,
        execution_market_view=execution_market_view,
    )
    return candidate_signals_df.merge(
        counterfactual_df,
        on="signal_id",
        how="left",
        validate="one_to_one",
    )


def _build_disabled_scored_signals(
    candidate_signals_df: pd.DataFrame, score_source: str
) -> pd.DataFrame:
    if candidate_signals_df.empty:
        return pd.DataFrame(columns=list(_SCORED_OUTPUT_COLUMNS))
    out = candidate_signals_df.copy()
    out["score_source"] = score_source
    out["p_block"] = 0.0
    outcome = (
        out.get("counterfactual_trade_outcome", pd.Series(pd.NA, index=out.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["target_block_signal"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    sl_mask = outcome.eq("sl")
    tp_mask = outcome.eq("tp")
    out.loc[sl_mask, "target_block_signal"] = 1
    out.loc[tp_mask, "target_block_signal"] = 0
    out["block_reason"] = outcome.apply(_resolve_execution_block_reason)
    out["gate_trainable_signal"] = False
    for column in _SCORED_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(_SCORED_OUTPUT_COLUMNS)].reset_index(drop=True)


def _resolve_execution_block_reason(outcome_value: object) -> str:
    outcome = str(outcome_value).strip().lower()
    if outcome == "sl":
        return "block_sl"
    if outcome == "tp":
        return "keep_tp"
    if outcome == "timeout":
        return "skip_timeout"
    if outcome == "ambiguous":
        return "skip_ambiguous"
    return "skip_unknown"


def _attach_counterfactual_columns(
    scored_signals_df: pd.DataFrame, enriched_candidate_signals_df: pd.DataFrame | None
) -> pd.DataFrame:
    if enriched_candidate_signals_df is None or enriched_candidate_signals_df.empty:
        out = scored_signals_df.copy()
        out["counterfactual_execution_status"] = pd.NA
        out["counterfactual_trade_outcome"] = pd.NA
        out["counterfactual_trade_pnl_pct"] = pd.NA
        out["counterfactual_exit_time"] = pd.NaT
        return out
    counterfactual_df = enriched_candidate_signals_df.loc[
        :,
        [
            "signal_id",
            "counterfactual_execution_status",
            "counterfactual_trade_outcome",
            "counterfactual_trade_pnl_pct",
            "counterfactual_exit_time",
        ],
    ].copy()
    return scored_signals_df.merge(
        counterfactual_df, on="signal_id", how="left", validate="one_to_one"
    )
