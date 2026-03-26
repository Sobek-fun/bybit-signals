from __future__ import annotations

import time

import pandas as pd

from pump_end_v2.features.manifest import (DETECTOR_FEATURE_COLUMNS,
                                           DETECTOR_IDENTITY_COLUMNS)
from pump_end_v2.logging import log_info, stage_done, stage_start


def build_detector_feature_view(
    decision_rows_df: pd.DataFrame,
    token_state_df: pd.DataFrame,
    reference_state_df: pd.DataFrame,
    breadth_state_df: pd.DataFrame,
    episode_state_df: pd.DataFrame,
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("FEATURES", "DETECTOR_VIEW")
    base = decision_rows_df.copy()
    token_cols = token_state_df[
        [
            "symbol",
            "open_time",
            "close_ret_1",
            "close_ret_4",
            "close_ret_12",
            "intrabar_range_pct",
            "candle_body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "rolling_volatility_4",
            "rolling_volatility_12",
            "rsi_like_14",
            "mfi_like_14",
            "macd_line",
            "macd_hist",
            "heat_flag",
            "fade_flag",
            "dollar_volume",
            "dollar_volume_ratio_12",
            "liquidity_score_12",
        ]
    ].rename(columns={"open_time": "context_bar_open_time"})
    frame = base.merge(token_cols, on=["symbol", "context_bar_open_time"], how="left")
    frame = frame.merge(
        reference_state_df,
        left_on="context_bar_open_time",
        right_on="open_time",
        how="left",
    ).drop(columns=["open_time"], errors="ignore")
    frame = frame.merge(
        breadth_state_df,
        left_on="context_bar_open_time",
        right_on="open_time",
        how="left",
    ).drop(columns=["open_time"], errors="ignore")
    frame = frame.merge(
        episode_state_df[
            [
                "decision_row_id",
                "episode_runup_from_open_pct",
                "episode_extension_from_open_pct",
                "bars_since_episode_high",
                "drawdown_from_episode_high_so_far",
                "high_retest_count",
                "high_persistence_4",
                "episode_pump_context_streak",
            ]
        ],
        on="decision_row_id",
        how="left",
    )
    ordered = frame.loc[
        :, [*DETECTOR_IDENTITY_COLUMNS, *DETECTOR_FEATURE_COLUMNS]
    ].copy()
    log_info(
        "FEATURES",
        f"detector_view summary rows_total={len(ordered)} feature_cols_total={len(DETECTOR_FEATURE_COLUMNS)}",
    )
    stage_done("FEATURES", "DETECTOR_VIEW", elapsed_sec=time.perf_counter() - started)
    return ordered
