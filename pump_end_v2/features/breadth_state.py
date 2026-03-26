from __future__ import annotations

import time

import pandas as pd

from pump_end_v2.logging import log_info, stage_done, stage_start

BREADTH_STATE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "breadth_universe_size",
    "breadth_advancers_share",
    "breadth_mean_close_ret_1",
    "breadth_median_close_ret_1",
    "breadth_std_close_ret_1",
    "breadth_near_high_share",
    "breadth_pump_context_share",
    "breadth_volume_spike_share",
)


def build_breadth_state_layer(
    token_state_df: pd.DataFrame, btc_symbol: str, eth_symbol: str
) -> pd.DataFrame:
    started = time.perf_counter()
    stage_start("LAYERS", "BREADTH_STATE")
    base_times = (
        token_state_df[["open_time"]]
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time", kind="mergesort")
        .reset_index(drop=True)
    )
    non_ref = token_state_df[
        ~token_state_df["symbol"].isin([btc_symbol, eth_symbol])
    ].copy()
    if non_ref.empty:
        breadth = base_times.assign(
            breadth_universe_size=0,
            breadth_advancers_share=0.0,
            breadth_mean_close_ret_1=0.0,
            breadth_median_close_ret_1=0.0,
            breadth_std_close_ret_1=0.0,
            breadth_near_high_share=0.0,
            breadth_pump_context_share=0.0,
            breadth_volume_spike_share=0.0,
        )
    else:
        grouped = non_ref.groupby("open_time", sort=True)
        aggregated = grouped.agg(
            breadth_universe_size=("symbol", "count"),
            breadth_advancers_share=("close_ret_1", lambda s: float((s > 0).mean())),
            breadth_mean_close_ret_1=("close_ret_1", "mean"),
            breadth_median_close_ret_1=("close_ret_1", "median"),
            breadth_std_close_ret_1=("close_ret_1", lambda s: float(s.std(ddof=0))),
            breadth_near_high_share=(
                "near_high_flag",
                lambda s: float(pd.Series(s).astype(bool).mean()),
            ),
            breadth_pump_context_share=(
                "pump_context_flag",
                lambda s: float(pd.Series(s).astype(bool).mean()),
            ),
            breadth_volume_spike_share=(
                "volume_ratio",
                lambda s: float((s > 1.0).mean()),
            ),
        ).reset_index()
        breadth = base_times.merge(aggregated, on="open_time", how="left")
        breadth["breadth_universe_size"] = (
            breadth["breadth_universe_size"].fillna(0).astype(int)
        )
        breadth["breadth_advancers_share"] = breadth["breadth_advancers_share"].fillna(
            0.0
        )
        breadth["breadth_mean_close_ret_1"] = breadth[
            "breadth_mean_close_ret_1"
        ].fillna(0.0)
        breadth["breadth_median_close_ret_1"] = breadth[
            "breadth_median_close_ret_1"
        ].fillna(0.0)
        breadth["breadth_std_close_ret_1"] = breadth["breadth_std_close_ret_1"].fillna(
            0.0
        )
        breadth["breadth_near_high_share"] = breadth["breadth_near_high_share"].fillna(
            0.0
        )
        breadth["breadth_pump_context_share"] = breadth[
            "breadth_pump_context_share"
        ].fillna(0.0)
        breadth["breadth_volume_spike_share"] = breadth[
            "breadth_volume_spike_share"
        ].fillna(0.0)
    breadth = breadth.loc[:, list(BREADTH_STATE_COLUMNS)].copy()
    log_info(
        "LAYERS",
        f"breadth_state summary timestamps_total={len(breadth)} cols_total={len(breadth.columns)}",
    )
    stage_done("LAYERS", "BREADTH_STATE", elapsed_sec=time.perf_counter() - started)
    return breadth
