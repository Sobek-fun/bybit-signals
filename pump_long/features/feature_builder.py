import numpy as np
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

from pump_long.infra.clickhouse import DataLoader
from pump_long.features.params import PumpParams, DEFAULT_PUMP_PARAMS


FEATURE_SETS = {
    "base": [
        "ret_1_lag_0", "ret_1_lag_1", "ret_1_lag_2", "ret_1_lag_3", "ret_1_lag_4",
        "ret_1_lag_5", "ret_1_lag_6", "ret_1_lag_7", "ret_1_lag_8", "ret_1_lag_9",
        "ret_1_lag_10", "ret_1_lag_11", "ret_1_lag_12", "ret_1_lag_13", "ret_1_lag_14",
        "ret_1_lag_15", "ret_1_lag_16", "ret_1_lag_17", "ret_1_lag_18", "ret_1_lag_19",
        "ret_1_lag_20", "ret_1_lag_21", "ret_1_lag_22", "ret_1_lag_23", "ret_1_lag_24",
        "ret_1_lag_25", "ret_1_lag_26", "ret_1_lag_27", "ret_1_lag_28", "ret_1_lag_29",
        "ret_1_lag_30", "ret_1_lag_31", "ret_1_lag_32", "ret_1_lag_33", "ret_1_lag_34",
        "ret_1_lag_35", "ret_1_lag_36", "ret_1_lag_37", "ret_1_lag_38", "ret_1_lag_39",
        "ret_1_lag_40", "ret_1_lag_41", "ret_1_lag_42", "ret_1_lag_43", "ret_1_lag_44",
        "ret_1_lag_45", "ret_1_lag_46", "ret_1_lag_47", "ret_1_lag_48", "ret_1_lag_49",
        "ret_1_lag_50", "ret_1_lag_51", "ret_1_lag_52", "ret_1_lag_53", "ret_1_lag_54",
        "ret_1_lag_55", "ret_1_lag_56", "ret_1_lag_57", "ret_1_lag_58", "ret_1_lag_59",
        "vol_ratio_lag_0", "vol_ratio_lag_1", "vol_ratio_lag_2", "vol_ratio_lag_3", "vol_ratio_lag_4",
        "vol_ratio_lag_5", "vol_ratio_lag_6", "vol_ratio_lag_7", "vol_ratio_lag_8", "vol_ratio_lag_9",
        "vol_ratio_lag_10", "vol_ratio_lag_11", "vol_ratio_lag_12", "vol_ratio_lag_13", "vol_ratio_lag_14",
        "vol_ratio_lag_15", "vol_ratio_lag_16", "vol_ratio_lag_17", "vol_ratio_lag_18", "vol_ratio_lag_19",
        "vol_ratio_lag_20", "vol_ratio_lag_21", "vol_ratio_lag_22", "vol_ratio_lag_23", "vol_ratio_lag_24",
        "vol_ratio_lag_25", "vol_ratio_lag_26", "vol_ratio_lag_27", "vol_ratio_lag_28", "vol_ratio_lag_29",
        "vol_ratio_lag_30", "vol_ratio_lag_31", "vol_ratio_lag_32", "vol_ratio_lag_33", "vol_ratio_lag_34",
        "vol_ratio_lag_35", "vol_ratio_lag_36", "vol_ratio_lag_37", "vol_ratio_lag_38", "vol_ratio_lag_39",
        "vol_ratio_lag_40", "vol_ratio_lag_41", "vol_ratio_lag_42", "vol_ratio_lag_43", "vol_ratio_lag_44",
        "vol_ratio_lag_45", "vol_ratio_lag_46", "vol_ratio_lag_47", "vol_ratio_lag_48", "vol_ratio_lag_49",
        "vol_ratio_lag_50", "vol_ratio_lag_51", "vol_ratio_lag_52", "vol_ratio_lag_53", "vol_ratio_lag_54",
        "vol_ratio_lag_55", "vol_ratio_lag_56", "vol_ratio_lag_57", "vol_ratio_lag_58", "vol_ratio_lag_59",
        "upper_wick_ratio_lag_0", "upper_wick_ratio_lag_1", "upper_wick_ratio_lag_2", "upper_wick_ratio_lag_3", "upper_wick_ratio_lag_4",
        "upper_wick_ratio_lag_5", "upper_wick_ratio_lag_6", "upper_wick_ratio_lag_7", "upper_wick_ratio_lag_8", "upper_wick_ratio_lag_9",
        "upper_wick_ratio_lag_10", "upper_wick_ratio_lag_11", "upper_wick_ratio_lag_12", "upper_wick_ratio_lag_13", "upper_wick_ratio_lag_14",
        "upper_wick_ratio_lag_15", "upper_wick_ratio_lag_16", "upper_wick_ratio_lag_17", "upper_wick_ratio_lag_18", "upper_wick_ratio_lag_19",
        "upper_wick_ratio_lag_20", "upper_wick_ratio_lag_21", "upper_wick_ratio_lag_22", "upper_wick_ratio_lag_23", "upper_wick_ratio_lag_24",
        "upper_wick_ratio_lag_25", "upper_wick_ratio_lag_26", "upper_wick_ratio_lag_27", "upper_wick_ratio_lag_28", "upper_wick_ratio_lag_29",
        "upper_wick_ratio_lag_30", "upper_wick_ratio_lag_31", "upper_wick_ratio_lag_32", "upper_wick_ratio_lag_33", "upper_wick_ratio_lag_34",
        "upper_wick_ratio_lag_35", "upper_wick_ratio_lag_36", "upper_wick_ratio_lag_37", "upper_wick_ratio_lag_38", "upper_wick_ratio_lag_39",
        "upper_wick_ratio_lag_40", "upper_wick_ratio_lag_41", "upper_wick_ratio_lag_42", "upper_wick_ratio_lag_43", "upper_wick_ratio_lag_44",
        "upper_wick_ratio_lag_45", "upper_wick_ratio_lag_46", "upper_wick_ratio_lag_47", "upper_wick_ratio_lag_48", "upper_wick_ratio_lag_49",
        "upper_wick_ratio_lag_50", "upper_wick_ratio_lag_51", "upper_wick_ratio_lag_52", "upper_wick_ratio_lag_53", "upper_wick_ratio_lag_54",
        "upper_wick_ratio_lag_55", "upper_wick_ratio_lag_56", "upper_wick_ratio_lag_57", "upper_wick_ratio_lag_58", "upper_wick_ratio_lag_59",
        "lower_wick_ratio_lag_0", "lower_wick_ratio_lag_1", "lower_wick_ratio_lag_2", "lower_wick_ratio_lag_3", "lower_wick_ratio_lag_4",
        "lower_wick_ratio_lag_5", "lower_wick_ratio_lag_6", "lower_wick_ratio_lag_7", "lower_wick_ratio_lag_8", "lower_wick_ratio_lag_9",
        "lower_wick_ratio_lag_10", "lower_wick_ratio_lag_11", "lower_wick_ratio_lag_12", "lower_wick_ratio_lag_13", "lower_wick_ratio_lag_14",
        "lower_wick_ratio_lag_15", "lower_wick_ratio_lag_16", "lower_wick_ratio_lag_17", "lower_wick_ratio_lag_18", "lower_wick_ratio_lag_19",
        "lower_wick_ratio_lag_20", "lower_wick_ratio_lag_21", "lower_wick_ratio_lag_22", "lower_wick_ratio_lag_23", "lower_wick_ratio_lag_24",
        "lower_wick_ratio_lag_25", "lower_wick_ratio_lag_26", "lower_wick_ratio_lag_27", "lower_wick_ratio_lag_28", "lower_wick_ratio_lag_29",
        "lower_wick_ratio_lag_30", "lower_wick_ratio_lag_31", "lower_wick_ratio_lag_32", "lower_wick_ratio_lag_33", "lower_wick_ratio_lag_34",
        "lower_wick_ratio_lag_35", "lower_wick_ratio_lag_36", "lower_wick_ratio_lag_37", "lower_wick_ratio_lag_38", "lower_wick_ratio_lag_39",
        "lower_wick_ratio_lag_40", "lower_wick_ratio_lag_41", "lower_wick_ratio_lag_42", "lower_wick_ratio_lag_43", "lower_wick_ratio_lag_44",
        "lower_wick_ratio_lag_45", "lower_wick_ratio_lag_46", "lower_wick_ratio_lag_47", "lower_wick_ratio_lag_48", "lower_wick_ratio_lag_49",
        "lower_wick_ratio_lag_50", "lower_wick_ratio_lag_51", "lower_wick_ratio_lag_52", "lower_wick_ratio_lag_53", "lower_wick_ratio_lag_54",
        "lower_wick_ratio_lag_55", "lower_wick_ratio_lag_56", "lower_wick_ratio_lag_57", "lower_wick_ratio_lag_58", "lower_wick_ratio_lag_59",
        "body_ratio_lag_0", "body_ratio_lag_1", "body_ratio_lag_2", "body_ratio_lag_3", "body_ratio_lag_4",
        "body_ratio_lag_5", "body_ratio_lag_6", "body_ratio_lag_7", "body_ratio_lag_8", "body_ratio_lag_9",
        "body_ratio_lag_10", "body_ratio_lag_11", "body_ratio_lag_12", "body_ratio_lag_13", "body_ratio_lag_14",
        "body_ratio_lag_15", "body_ratio_lag_16", "body_ratio_lag_17", "body_ratio_lag_18", "body_ratio_lag_19",
        "body_ratio_lag_20", "body_ratio_lag_21", "body_ratio_lag_22", "body_ratio_lag_23", "body_ratio_lag_24",
        "body_ratio_lag_25", "body_ratio_lag_26", "body_ratio_lag_27", "body_ratio_lag_28", "body_ratio_lag_29",
        "body_ratio_lag_30", "body_ratio_lag_31", "body_ratio_lag_32", "body_ratio_lag_33", "body_ratio_lag_34",
        "body_ratio_lag_35", "body_ratio_lag_36", "body_ratio_lag_37", "body_ratio_lag_38", "body_ratio_lag_39",
        "body_ratio_lag_40", "body_ratio_lag_41", "body_ratio_lag_42", "body_ratio_lag_43", "body_ratio_lag_44",
        "body_ratio_lag_45", "body_ratio_lag_46", "body_ratio_lag_47", "body_ratio_lag_48", "body_ratio_lag_49",
        "body_ratio_lag_50", "body_ratio_lag_51", "body_ratio_lag_52", "body_ratio_lag_53", "body_ratio_lag_54",
        "body_ratio_lag_55", "body_ratio_lag_56", "body_ratio_lag_57", "body_ratio_lag_58", "body_ratio_lag_59",
        "range_lag_0", "range_lag_1", "range_lag_2", "range_lag_3", "range_lag_4",
        "range_lag_5", "range_lag_6", "range_lag_7", "range_lag_8", "range_lag_9",
        "range_lag_10", "range_lag_11", "range_lag_12", "range_lag_13", "range_lag_14",
        "range_lag_15", "range_lag_16", "range_lag_17", "range_lag_18", "range_lag_19",
        "range_lag_20", "range_lag_21", "range_lag_22", "range_lag_23", "range_lag_24",
        "range_lag_25", "range_lag_26", "range_lag_27", "range_lag_28", "range_lag_29",
        "range_lag_30", "range_lag_31", "range_lag_32", "range_lag_33", "range_lag_34",
        "range_lag_35", "range_lag_36", "range_lag_37", "range_lag_38", "range_lag_39",
        "range_lag_40", "range_lag_41", "range_lag_42", "range_lag_43", "range_lag_44",
        "range_lag_45", "range_lag_46", "range_lag_47", "range_lag_48", "range_lag_49",
        "range_lag_50", "range_lag_51", "range_lag_52", "range_lag_53", "range_lag_54",
        "range_lag_55", "range_lag_56", "range_lag_57", "range_lag_58", "range_lag_59",
        "close_pos_lag_0", "close_pos_lag_1", "close_pos_lag_2", "close_pos_lag_3", "close_pos_lag_4",
        "close_pos_lag_5", "close_pos_lag_6", "close_pos_lag_7", "close_pos_lag_8", "close_pos_lag_9",
        "close_pos_lag_10", "close_pos_lag_11", "close_pos_lag_12", "close_pos_lag_13", "close_pos_lag_14",
        "close_pos_lag_15", "close_pos_lag_16", "close_pos_lag_17", "close_pos_lag_18", "close_pos_lag_19",
        "close_pos_lag_20", "close_pos_lag_21", "close_pos_lag_22", "close_pos_lag_23", "close_pos_lag_24",
        "close_pos_lag_25", "close_pos_lag_26", "close_pos_lag_27", "close_pos_lag_28", "close_pos_lag_29",
        "close_pos_lag_30", "close_pos_lag_31", "close_pos_lag_32", "close_pos_lag_33", "close_pos_lag_34",
        "close_pos_lag_35", "close_pos_lag_36", "close_pos_lag_37", "close_pos_lag_38", "close_pos_lag_39",
        "close_pos_lag_40", "close_pos_lag_41", "close_pos_lag_42", "close_pos_lag_43", "close_pos_lag_44",
        "close_pos_lag_45", "close_pos_lag_46", "close_pos_lag_47", "close_pos_lag_48", "close_pos_lag_49",
        "close_pos_lag_50", "close_pos_lag_51", "close_pos_lag_52", "close_pos_lag_53", "close_pos_lag_54",
        "close_pos_lag_55", "close_pos_lag_56", "close_pos_lag_57", "close_pos_lag_58", "close_pos_lag_59",
        "wick_ratio_lag_0", "wick_ratio_lag_1", "wick_ratio_lag_2", "wick_ratio_lag_3", "wick_ratio_lag_4",
        "wick_ratio_lag_5", "wick_ratio_lag_6", "wick_ratio_lag_7", "wick_ratio_lag_8", "wick_ratio_lag_9",
        "wick_ratio_lag_10", "wick_ratio_lag_11", "wick_ratio_lag_12", "wick_ratio_lag_13", "wick_ratio_lag_14",
        "wick_ratio_lag_15", "wick_ratio_lag_16", "wick_ratio_lag_17", "wick_ratio_lag_18", "wick_ratio_lag_19",
        "wick_ratio_lag_20", "wick_ratio_lag_21", "wick_ratio_lag_22", "wick_ratio_lag_23", "wick_ratio_lag_24",
        "wick_ratio_lag_25", "wick_ratio_lag_26", "wick_ratio_lag_27", "wick_ratio_lag_28", "wick_ratio_lag_29",
        "wick_ratio_lag_30", "wick_ratio_lag_31", "wick_ratio_lag_32", "wick_ratio_lag_33", "wick_ratio_lag_34",
        "wick_ratio_lag_35", "wick_ratio_lag_36", "wick_ratio_lag_37", "wick_ratio_lag_38", "wick_ratio_lag_39",
        "wick_ratio_lag_40", "wick_ratio_lag_41", "wick_ratio_lag_42", "wick_ratio_lag_43", "wick_ratio_lag_44",
        "wick_ratio_lag_45", "wick_ratio_lag_46", "wick_ratio_lag_47", "wick_ratio_lag_48", "wick_ratio_lag_49",
        "wick_ratio_lag_50", "wick_ratio_lag_51", "wick_ratio_lag_52", "wick_ratio_lag_53", "wick_ratio_lag_54",
        "wick_ratio_lag_55", "wick_ratio_lag_56", "wick_ratio_lag_57", "wick_ratio_lag_58", "wick_ratio_lag_59",
        "liq_sweep_flag_lag_0", "liq_sweep_flag_lag_1", "liq_sweep_flag_lag_2", "liq_sweep_flag_lag_3", "liq_sweep_flag_lag_4",
        "liq_sweep_flag_lag_5", "liq_sweep_flag_lag_6", "liq_sweep_flag_lag_7", "liq_sweep_flag_lag_8", "liq_sweep_flag_lag_9",
        "liq_sweep_flag_lag_10", "liq_sweep_flag_lag_11", "liq_sweep_flag_lag_12", "liq_sweep_flag_lag_13", "liq_sweep_flag_lag_14",
        "liq_sweep_flag_lag_15", "liq_sweep_flag_lag_16", "liq_sweep_flag_lag_17", "liq_sweep_flag_lag_18", "liq_sweep_flag_lag_19",
        "liq_sweep_flag_lag_20", "liq_sweep_flag_lag_21", "liq_sweep_flag_lag_22", "liq_sweep_flag_lag_23", "liq_sweep_flag_lag_24",
        "liq_sweep_flag_lag_25", "liq_sweep_flag_lag_26", "liq_sweep_flag_lag_27", "liq_sweep_flag_lag_28", "liq_sweep_flag_lag_29",
        "liq_sweep_flag_lag_30", "liq_sweep_flag_lag_31", "liq_sweep_flag_lag_32", "liq_sweep_flag_lag_33", "liq_sweep_flag_lag_34",
        "liq_sweep_flag_lag_35", "liq_sweep_flag_lag_36", "liq_sweep_flag_lag_37", "liq_sweep_flag_lag_38", "liq_sweep_flag_lag_39",
        "liq_sweep_flag_lag_40", "liq_sweep_flag_lag_41", "liq_sweep_flag_lag_42", "liq_sweep_flag_lag_43", "liq_sweep_flag_lag_44",
        "liq_sweep_flag_lag_45", "liq_sweep_flag_lag_46", "liq_sweep_flag_lag_47", "liq_sweep_flag_lag_48", "liq_sweep_flag_lag_49",
        "liq_sweep_flag_lag_50", "liq_sweep_flag_lag_51", "liq_sweep_flag_lag_52", "liq_sweep_flag_lag_53", "liq_sweep_flag_lag_54",
        "liq_sweep_flag_lag_55", "liq_sweep_flag_lag_56", "liq_sweep_flag_lag_57", "liq_sweep_flag_lag_58", "liq_sweep_flag_lag_59",
        "liq_sweep_overshoot_lag_0", "liq_sweep_overshoot_lag_1", "liq_sweep_overshoot_lag_2", "liq_sweep_overshoot_lag_3", "liq_sweep_overshoot_lag_4",
        "liq_sweep_overshoot_lag_5", "liq_sweep_overshoot_lag_6", "liq_sweep_overshoot_lag_7", "liq_sweep_overshoot_lag_8", "liq_sweep_overshoot_lag_9",
        "liq_sweep_overshoot_lag_10", "liq_sweep_overshoot_lag_11", "liq_sweep_overshoot_lag_12", "liq_sweep_overshoot_lag_13", "liq_sweep_overshoot_lag_14",
        "liq_sweep_overshoot_lag_15", "liq_sweep_overshoot_lag_16", "liq_sweep_overshoot_lag_17", "liq_sweep_overshoot_lag_18", "liq_sweep_overshoot_lag_19",
        "liq_sweep_overshoot_lag_20", "liq_sweep_overshoot_lag_21", "liq_sweep_overshoot_lag_22", "liq_sweep_overshoot_lag_23", "liq_sweep_overshoot_lag_24",
        "liq_sweep_overshoot_lag_25", "liq_sweep_overshoot_lag_26", "liq_sweep_overshoot_lag_27", "liq_sweep_overshoot_lag_28", "liq_sweep_overshoot_lag_29",
        "liq_sweep_overshoot_lag_30", "liq_sweep_overshoot_lag_31", "liq_sweep_overshoot_lag_32", "liq_sweep_overshoot_lag_33", "liq_sweep_overshoot_lag_34",
        "liq_sweep_overshoot_lag_35", "liq_sweep_overshoot_lag_36", "liq_sweep_overshoot_lag_37", "liq_sweep_overshoot_lag_38", "liq_sweep_overshoot_lag_39",
        "liq_sweep_overshoot_lag_40", "liq_sweep_overshoot_lag_41", "liq_sweep_overshoot_lag_42", "liq_sweep_overshoot_lag_43", "liq_sweep_overshoot_lag_44",
        "liq_sweep_overshoot_lag_45", "liq_sweep_overshoot_lag_46", "liq_sweep_overshoot_lag_47", "liq_sweep_overshoot_lag_48", "liq_sweep_overshoot_lag_49",
        "liq_sweep_overshoot_lag_50", "liq_sweep_overshoot_lag_51", "liq_sweep_overshoot_lag_52", "liq_sweep_overshoot_lag_53", "liq_sweep_overshoot_lag_54",
        "liq_sweep_overshoot_lag_55", "liq_sweep_overshoot_lag_56", "liq_sweep_overshoot_lag_57", "liq_sweep_overshoot_lag_58", "liq_sweep_overshoot_lag_59",
        "liq_reject_strength_lag_0", "liq_reject_strength_lag_1", "liq_reject_strength_lag_2", "liq_reject_strength_lag_3", "liq_reject_strength_lag_4",
        "liq_reject_strength_lag_5", "liq_reject_strength_lag_6", "liq_reject_strength_lag_7", "liq_reject_strength_lag_8", "liq_reject_strength_lag_9",
        "liq_reject_strength_lag_10", "liq_reject_strength_lag_11", "liq_reject_strength_lag_12", "liq_reject_strength_lag_13", "liq_reject_strength_lag_14",
        "liq_reject_strength_lag_15", "liq_reject_strength_lag_16", "liq_reject_strength_lag_17", "liq_reject_strength_lag_18", "liq_reject_strength_lag_19",
        "liq_reject_strength_lag_20", "liq_reject_strength_lag_21", "liq_reject_strength_lag_22", "liq_reject_strength_lag_23", "liq_reject_strength_lag_24",
        "liq_reject_strength_lag_25", "liq_reject_strength_lag_26", "liq_reject_strength_lag_27", "liq_reject_strength_lag_28", "liq_reject_strength_lag_29",
        "liq_reject_strength_lag_30", "liq_reject_strength_lag_31", "liq_reject_strength_lag_32", "liq_reject_strength_lag_33", "liq_reject_strength_lag_34",
        "liq_reject_strength_lag_35", "liq_reject_strength_lag_36", "liq_reject_strength_lag_37", "liq_reject_strength_lag_38", "liq_reject_strength_lag_39",
        "liq_reject_strength_lag_40", "liq_reject_strength_lag_41", "liq_reject_strength_lag_42", "liq_reject_strength_lag_43", "liq_reject_strength_lag_44",
        "liq_reject_strength_lag_45", "liq_reject_strength_lag_46", "liq_reject_strength_lag_47", "liq_reject_strength_lag_48", "liq_reject_strength_lag_49",
        "liq_reject_strength_lag_50", "liq_reject_strength_lag_51", "liq_reject_strength_lag_52", "liq_reject_strength_lag_53", "liq_reject_strength_lag_54",
        "liq_reject_strength_lag_55", "liq_reject_strength_lag_56", "liq_reject_strength_lag_57", "liq_reject_strength_lag_58", "liq_reject_strength_lag_59",
        "range_over_atr_lag_0", "range_over_atr_lag_1", "range_over_atr_lag_2", "range_over_atr_lag_3", "range_over_atr_lag_4",
        "range_over_atr_lag_5", "range_over_atr_lag_6", "range_over_atr_lag_7", "range_over_atr_lag_8", "range_over_atr_lag_9",
        "range_over_atr_lag_10", "range_over_atr_lag_11", "range_over_atr_lag_12", "range_over_atr_lag_13", "range_over_atr_lag_14",
        "range_over_atr_lag_15", "range_over_atr_lag_16", "range_over_atr_lag_17", "range_over_atr_lag_18", "range_over_atr_lag_19",
        "range_over_atr_lag_20", "range_over_atr_lag_21", "range_over_atr_lag_22", "range_over_atr_lag_23", "range_over_atr_lag_24",
        "range_over_atr_lag_25", "range_over_atr_lag_26", "range_over_atr_lag_27", "range_over_atr_lag_28", "range_over_atr_lag_29",
        "range_over_atr_lag_30", "range_over_atr_lag_31", "range_over_atr_lag_32", "range_over_atr_lag_33", "range_over_atr_lag_34",
        "range_over_atr_lag_35", "range_over_atr_lag_36", "range_over_atr_lag_37", "range_over_atr_lag_38", "range_over_atr_lag_39",
        "range_over_atr_lag_40", "range_over_atr_lag_41", "range_over_atr_lag_42", "range_over_atr_lag_43", "range_over_atr_lag_44",
        "range_over_atr_lag_45", "range_over_atr_lag_46", "range_over_atr_lag_47", "range_over_atr_lag_48", "range_over_atr_lag_49",
        "range_over_atr_lag_50", "range_over_atr_lag_51", "range_over_atr_lag_52", "range_over_atr_lag_53", "range_over_atr_lag_54",
        "range_over_atr_lag_55", "range_over_atr_lag_56", "range_over_atr_lag_57", "range_over_atr_lag_58", "range_over_atr_lag_59",
        "upper_wick_over_atr_lag_0", "upper_wick_over_atr_lag_1", "upper_wick_over_atr_lag_2", "upper_wick_over_atr_lag_3", "upper_wick_over_atr_lag_4",
        "upper_wick_over_atr_lag_5", "upper_wick_over_atr_lag_6", "upper_wick_over_atr_lag_7", "upper_wick_over_atr_lag_8", "upper_wick_over_atr_lag_9",
        "upper_wick_over_atr_lag_10", "upper_wick_over_atr_lag_11", "upper_wick_over_atr_lag_12", "upper_wick_over_atr_lag_13", "upper_wick_over_atr_lag_14",
        "upper_wick_over_atr_lag_15", "upper_wick_over_atr_lag_16", "upper_wick_over_atr_lag_17", "upper_wick_over_atr_lag_18", "upper_wick_over_atr_lag_19",
        "upper_wick_over_atr_lag_20", "upper_wick_over_atr_lag_21", "upper_wick_over_atr_lag_22", "upper_wick_over_atr_lag_23", "upper_wick_over_atr_lag_24",
        "upper_wick_over_atr_lag_25", "upper_wick_over_atr_lag_26", "upper_wick_over_atr_lag_27", "upper_wick_over_atr_lag_28", "upper_wick_over_atr_lag_29",
        "upper_wick_over_atr_lag_30", "upper_wick_over_atr_lag_31", "upper_wick_over_atr_lag_32", "upper_wick_over_atr_lag_33", "upper_wick_over_atr_lag_34",
        "upper_wick_over_atr_lag_35", "upper_wick_over_atr_lag_36", "upper_wick_over_atr_lag_37", "upper_wick_over_atr_lag_38", "upper_wick_over_atr_lag_39",
        "upper_wick_over_atr_lag_40", "upper_wick_over_atr_lag_41", "upper_wick_over_atr_lag_42", "upper_wick_over_atr_lag_43", "upper_wick_over_atr_lag_44",
        "upper_wick_over_atr_lag_45", "upper_wick_over_atr_lag_46", "upper_wick_over_atr_lag_47", "upper_wick_over_atr_lag_48", "upper_wick_over_atr_lag_49",
        "upper_wick_over_atr_lag_50", "upper_wick_over_atr_lag_51", "upper_wick_over_atr_lag_52", "upper_wick_over_atr_lag_53", "upper_wick_over_atr_lag_54",
        "upper_wick_over_atr_lag_55", "upper_wick_over_atr_lag_56", "upper_wick_over_atr_lag_57", "upper_wick_over_atr_lag_58", "upper_wick_over_atr_lag_59",
        "signed_body_lag_0", "signed_body_lag_1", "signed_body_lag_2", "signed_body_lag_3", "signed_body_lag_4",
        "signed_body_lag_5", "signed_body_lag_6", "signed_body_lag_7", "signed_body_lag_8", "signed_body_lag_9",
        "signed_body_lag_10", "signed_body_lag_11", "signed_body_lag_12", "signed_body_lag_13", "signed_body_lag_14",
        "signed_body_lag_15", "signed_body_lag_16", "signed_body_lag_17", "signed_body_lag_18", "signed_body_lag_19",
        "signed_body_lag_20", "signed_body_lag_21", "signed_body_lag_22", "signed_body_lag_23", "signed_body_lag_24",
        "signed_body_lag_25", "signed_body_lag_26", "signed_body_lag_27", "signed_body_lag_28", "signed_body_lag_29",
        "signed_body_lag_30", "signed_body_lag_31", "signed_body_lag_32", "signed_body_lag_33", "signed_body_lag_34",
        "signed_body_lag_35", "signed_body_lag_36", "signed_body_lag_37", "signed_body_lag_38", "signed_body_lag_39",
        "signed_body_lag_40", "signed_body_lag_41", "signed_body_lag_42", "signed_body_lag_43", "signed_body_lag_44",
        "signed_body_lag_45", "signed_body_lag_46", "signed_body_lag_47", "signed_body_lag_48", "signed_body_lag_49",
        "signed_body_lag_50", "signed_body_lag_51", "signed_body_lag_52", "signed_body_lag_53", "signed_body_lag_54",
        "signed_body_lag_55", "signed_body_lag_56", "signed_body_lag_57", "signed_body_lag_58", "signed_body_lag_59",
        "climax_vr_lag_0", "climax_vr_lag_1", "climax_vr_lag_2", "climax_vr_lag_3", "climax_vr_lag_4",
        "climax_vr_lag_5", "climax_vr_lag_6", "climax_vr_lag_7", "climax_vr_lag_8", "climax_vr_lag_9",
        "climax_vr_lag_10", "climax_vr_lag_11", "climax_vr_lag_12", "climax_vr_lag_13", "climax_vr_lag_14",
        "climax_vr_lag_15", "climax_vr_lag_16", "climax_vr_lag_17", "climax_vr_lag_18", "climax_vr_lag_19",
        "climax_vr_lag_20", "climax_vr_lag_21", "climax_vr_lag_22", "climax_vr_lag_23", "climax_vr_lag_24",
        "climax_vr_lag_25", "climax_vr_lag_26", "climax_vr_lag_27", "climax_vr_lag_28", "climax_vr_lag_29",
        "climax_vr_lag_30", "climax_vr_lag_31", "climax_vr_lag_32", "climax_vr_lag_33", "climax_vr_lag_34",
        "climax_vr_lag_35", "climax_vr_lag_36", "climax_vr_lag_37", "climax_vr_lag_38", "climax_vr_lag_39",
        "climax_vr_lag_40", "climax_vr_lag_41", "climax_vr_lag_42", "climax_vr_lag_43", "climax_vr_lag_44",
        "climax_vr_lag_45", "climax_vr_lag_46", "climax_vr_lag_47", "climax_vr_lag_48", "climax_vr_lag_49",
        "climax_vr_lag_50", "climax_vr_lag_51", "climax_vr_lag_52", "climax_vr_lag_53", "climax_vr_lag_54",
        "climax_vr_lag_55", "climax_vr_lag_56", "climax_vr_lag_57", "climax_vr_lag_58", "climax_vr_lag_59",
        "ret_accel_lag_0", "ret_accel_lag_1", "ret_accel_lag_2", "ret_accel_lag_3", "ret_accel_lag_4",
        "ret_accel_lag_5", "ret_accel_lag_6", "ret_accel_lag_7", "ret_accel_lag_8", "ret_accel_lag_9",
        "ret_accel_lag_10", "ret_accel_lag_11", "ret_accel_lag_12", "ret_accel_lag_13", "ret_accel_lag_14",
        "ret_accel_lag_15", "ret_accel_lag_16", "ret_accel_lag_17", "ret_accel_lag_18", "ret_accel_lag_19",
        "ret_accel_lag_20", "ret_accel_lag_21", "ret_accel_lag_22", "ret_accel_lag_23", "ret_accel_lag_24",
        "ret_accel_lag_25", "ret_accel_lag_26", "ret_accel_lag_27", "ret_accel_lag_28", "ret_accel_lag_29",
        "ret_accel_lag_30", "ret_accel_lag_31", "ret_accel_lag_32", "ret_accel_lag_33", "ret_accel_lag_34",
        "ret_accel_lag_35", "ret_accel_lag_36", "ret_accel_lag_37", "ret_accel_lag_38", "ret_accel_lag_39",
        "ret_accel_lag_40", "ret_accel_lag_41", "ret_accel_lag_42", "ret_accel_lag_43", "ret_accel_lag_44",
        "ret_accel_lag_45", "ret_accel_lag_46", "ret_accel_lag_47", "ret_accel_lag_48", "ret_accel_lag_49",
        "ret_accel_lag_50", "ret_accel_lag_51", "ret_accel_lag_52", "ret_accel_lag_53", "ret_accel_lag_54",
        "ret_accel_lag_55", "ret_accel_lag_56", "ret_accel_lag_57", "ret_accel_lag_58", "ret_accel_lag_59",
        "rsi_14_max_60", "rsi_14_min_60", "rsi_14_mean_60", "rsi_14_std_60", "rsi_14_last_minus_max_60",
        "rsi_14_slope_5", "rsi_14_delta_1", "rsi_14_delta_3", "rsi_14_delta_5",
        "mfi_14_max_60", "mfi_14_min_60", "mfi_14_mean_60", "mfi_14_std_60", "mfi_14_last_minus_max_60",
        "mfi_14_slope_5", "mfi_14_delta_1", "mfi_14_delta_3", "mfi_14_delta_5",
        "macdh_12_26_9_max_60", "macdh_12_26_9_min_60", "macdh_12_26_9_mean_60", "macdh_12_26_9_std_60", "macdh_12_26_9_last_minus_max_60",
        "macdh_12_26_9_slope_5", "macdh_12_26_9_delta_1", "macdh_12_26_9_delta_3", "macdh_12_26_9_delta_5",
        "macd_line_max_60", "macd_line_min_60", "macd_line_mean_60", "macd_line_std_60", "macd_line_last_minus_max_60",
        "macd_line_slope_5", "macd_line_delta_1", "macd_line_delta_3", "macd_line_delta_5",
        "vol_ratio_max_60", "vol_ratio_min_60", "vol_ratio_mean_60", "vol_ratio_std_60", "vol_ratio_last_minus_max_60",
        "vol_ratio_slope_5", "vol_ratio_delta_1", "vol_ratio_delta_3", "vol_ratio_delta_5",
        "ret_1_max_60", "ret_1_min_60", "ret_1_mean_60", "ret_1_std_60", "ret_1_last_minus_max_60",
        "ret_1_slope_5", "ret_1_delta_1", "ret_1_delta_3", "ret_1_delta_5",
        "drawdown_max_60", "drawdown_min_60", "drawdown_mean_60", "drawdown_std_60", "drawdown_last_minus_max_60",
        "drawdown_slope_5", "drawdown_delta_1", "drawdown_delta_3", "drawdown_delta_5",
        "rsi_14_minus_corridor", "mfi_14_minus_corridor", "macdh_12_26_9_minus_corridor",
        "runup", "runup_met", "vol_spike_cond", "vol_spike_recent",
        "rsi_hot", "mfi_hot", "osc_hot_recent", "macd_pos_recent",
        "pump_ctx", "near_peak", "blowoff_exhaustion",
        "osc_extreme", "predump_mask", "vol_fade", "rsi_fade", "macd_fade",
        "predump_peak", "strong_cond", "pump_score",
        "dist_to_pdh", "dist_to_pwh", "dist_to_eqh",
        "touched_pdh", "touched_pwh", "sweep_pdh", "sweep_pwh", "sweep_eqh",
        "overshoot_pdh", "overshoot_pwh", "overshoot_eqh",
        "eqh_strength", "eqh_age_bars",
        "liq_level_type_pwh", "liq_level_type_pdh", "liq_level_type_eqh",
        "liq_level_dist",
        "cum_ret_5", "cum_ret_10", "cum_ret_60",
        "count_red_last_5", "max_upper_wick_last_5",
        "vol_ratio_max_10", "vol_ratio_slope_5", "volume_fade",
    ],
    "extended": [
        "ret_1_lag_0", "ret_1_lag_1", "ret_1_lag_2", "ret_1_lag_3", "ret_1_lag_4",
        "ret_1_lag_5", "ret_1_lag_6", "ret_1_lag_7", "ret_1_lag_8", "ret_1_lag_9",
        "ret_1_lag_10", "ret_1_lag_11", "ret_1_lag_12", "ret_1_lag_13", "ret_1_lag_14",
        "ret_1_lag_15", "ret_1_lag_16", "ret_1_lag_17", "ret_1_lag_18", "ret_1_lag_19",
        "ret_1_lag_20", "ret_1_lag_21", "ret_1_lag_22", "ret_1_lag_23", "ret_1_lag_24",
        "ret_1_lag_25", "ret_1_lag_26", "ret_1_lag_27", "ret_1_lag_28", "ret_1_lag_29",
        "ret_1_lag_30", "ret_1_lag_31", "ret_1_lag_32", "ret_1_lag_33", "ret_1_lag_34",
        "ret_1_lag_35", "ret_1_lag_36", "ret_1_lag_37", "ret_1_lag_38", "ret_1_lag_39",
        "ret_1_lag_40", "ret_1_lag_41", "ret_1_lag_42", "ret_1_lag_43", "ret_1_lag_44",
        "ret_1_lag_45", "ret_1_lag_46", "ret_1_lag_47", "ret_1_lag_48", "ret_1_lag_49",
        "ret_1_lag_50", "ret_1_lag_51", "ret_1_lag_52", "ret_1_lag_53", "ret_1_lag_54",
        "ret_1_lag_55", "ret_1_lag_56", "ret_1_lag_57", "ret_1_lag_58", "ret_1_lag_59",
        "vol_ratio_lag_0", "vol_ratio_lag_1", "vol_ratio_lag_2", "vol_ratio_lag_3", "vol_ratio_lag_4",
        "vol_ratio_lag_5", "vol_ratio_lag_6", "vol_ratio_lag_7", "vol_ratio_lag_8", "vol_ratio_lag_9",
        "vol_ratio_lag_10", "vol_ratio_lag_11", "vol_ratio_lag_12", "vol_ratio_lag_13", "vol_ratio_lag_14",
        "vol_ratio_lag_15", "vol_ratio_lag_16", "vol_ratio_lag_17", "vol_ratio_lag_18", "vol_ratio_lag_19",
        "vol_ratio_lag_20", "vol_ratio_lag_21", "vol_ratio_lag_22", "vol_ratio_lag_23", "vol_ratio_lag_24",
        "vol_ratio_lag_25", "vol_ratio_lag_26", "vol_ratio_lag_27", "vol_ratio_lag_28", "vol_ratio_lag_29",
        "vol_ratio_lag_30", "vol_ratio_lag_31", "vol_ratio_lag_32", "vol_ratio_lag_33", "vol_ratio_lag_34",
        "vol_ratio_lag_35", "vol_ratio_lag_36", "vol_ratio_lag_37", "vol_ratio_lag_38", "vol_ratio_lag_39",
        "vol_ratio_lag_40", "vol_ratio_lag_41", "vol_ratio_lag_42", "vol_ratio_lag_43", "vol_ratio_lag_44",
        "vol_ratio_lag_45", "vol_ratio_lag_46", "vol_ratio_lag_47", "vol_ratio_lag_48", "vol_ratio_lag_49",
        "vol_ratio_lag_50", "vol_ratio_lag_51", "vol_ratio_lag_52", "vol_ratio_lag_53", "vol_ratio_lag_54",
        "vol_ratio_lag_55", "vol_ratio_lag_56", "vol_ratio_lag_57", "vol_ratio_lag_58", "vol_ratio_lag_59",
        "upper_wick_ratio_lag_0", "upper_wick_ratio_lag_1", "upper_wick_ratio_lag_2", "upper_wick_ratio_lag_3", "upper_wick_ratio_lag_4",
        "upper_wick_ratio_lag_5", "upper_wick_ratio_lag_6", "upper_wick_ratio_lag_7", "upper_wick_ratio_lag_8", "upper_wick_ratio_lag_9",
        "upper_wick_ratio_lag_10", "upper_wick_ratio_lag_11", "upper_wick_ratio_lag_12", "upper_wick_ratio_lag_13", "upper_wick_ratio_lag_14",
        "upper_wick_ratio_lag_15", "upper_wick_ratio_lag_16", "upper_wick_ratio_lag_17", "upper_wick_ratio_lag_18", "upper_wick_ratio_lag_19",
        "upper_wick_ratio_lag_20", "upper_wick_ratio_lag_21", "upper_wick_ratio_lag_22", "upper_wick_ratio_lag_23", "upper_wick_ratio_lag_24",
        "upper_wick_ratio_lag_25", "upper_wick_ratio_lag_26", "upper_wick_ratio_lag_27", "upper_wick_ratio_lag_28", "upper_wick_ratio_lag_29",
        "upper_wick_ratio_lag_30", "upper_wick_ratio_lag_31", "upper_wick_ratio_lag_32", "upper_wick_ratio_lag_33", "upper_wick_ratio_lag_34",
        "upper_wick_ratio_lag_35", "upper_wick_ratio_lag_36", "upper_wick_ratio_lag_37", "upper_wick_ratio_lag_38", "upper_wick_ratio_lag_39",
        "upper_wick_ratio_lag_40", "upper_wick_ratio_lag_41", "upper_wick_ratio_lag_42", "upper_wick_ratio_lag_43", "upper_wick_ratio_lag_44",
        "upper_wick_ratio_lag_45", "upper_wick_ratio_lag_46", "upper_wick_ratio_lag_47", "upper_wick_ratio_lag_48", "upper_wick_ratio_lag_49",
        "upper_wick_ratio_lag_50", "upper_wick_ratio_lag_51", "upper_wick_ratio_lag_52", "upper_wick_ratio_lag_53", "upper_wick_ratio_lag_54",
        "upper_wick_ratio_lag_55", "upper_wick_ratio_lag_56", "upper_wick_ratio_lag_57", "upper_wick_ratio_lag_58", "upper_wick_ratio_lag_59",
        "lower_wick_ratio_lag_0", "lower_wick_ratio_lag_1", "lower_wick_ratio_lag_2", "lower_wick_ratio_lag_3", "lower_wick_ratio_lag_4",
        "lower_wick_ratio_lag_5", "lower_wick_ratio_lag_6", "lower_wick_ratio_lag_7", "lower_wick_ratio_lag_8", "lower_wick_ratio_lag_9",
        "lower_wick_ratio_lag_10", "lower_wick_ratio_lag_11", "lower_wick_ratio_lag_12", "lower_wick_ratio_lag_13", "lower_wick_ratio_lag_14",
        "lower_wick_ratio_lag_15", "lower_wick_ratio_lag_16", "lower_wick_ratio_lag_17", "lower_wick_ratio_lag_18", "lower_wick_ratio_lag_19",
        "lower_wick_ratio_lag_20", "lower_wick_ratio_lag_21", "lower_wick_ratio_lag_22", "lower_wick_ratio_lag_23", "lower_wick_ratio_lag_24",
        "lower_wick_ratio_lag_25", "lower_wick_ratio_lag_26", "lower_wick_ratio_lag_27", "lower_wick_ratio_lag_28", "lower_wick_ratio_lag_29",
        "lower_wick_ratio_lag_30", "lower_wick_ratio_lag_31", "lower_wick_ratio_lag_32", "lower_wick_ratio_lag_33", "lower_wick_ratio_lag_34",
        "lower_wick_ratio_lag_35", "lower_wick_ratio_lag_36", "lower_wick_ratio_lag_37", "lower_wick_ratio_lag_38", "lower_wick_ratio_lag_39",
        "lower_wick_ratio_lag_40", "lower_wick_ratio_lag_41", "lower_wick_ratio_lag_42", "lower_wick_ratio_lag_43", "lower_wick_ratio_lag_44",
        "lower_wick_ratio_lag_45", "lower_wick_ratio_lag_46", "lower_wick_ratio_lag_47", "lower_wick_ratio_lag_48", "lower_wick_ratio_lag_49",
        "lower_wick_ratio_lag_50", "lower_wick_ratio_lag_51", "lower_wick_ratio_lag_52", "lower_wick_ratio_lag_53", "lower_wick_ratio_lag_54",
        "lower_wick_ratio_lag_55", "lower_wick_ratio_lag_56", "lower_wick_ratio_lag_57", "lower_wick_ratio_lag_58", "lower_wick_ratio_lag_59",
        "body_ratio_lag_0", "body_ratio_lag_1", "body_ratio_lag_2", "body_ratio_lag_3", "body_ratio_lag_4",
        "body_ratio_lag_5", "body_ratio_lag_6", "body_ratio_lag_7", "body_ratio_lag_8", "body_ratio_lag_9",
        "body_ratio_lag_10", "body_ratio_lag_11", "body_ratio_lag_12", "body_ratio_lag_13", "body_ratio_lag_14",
        "body_ratio_lag_15", "body_ratio_lag_16", "body_ratio_lag_17", "body_ratio_lag_18", "body_ratio_lag_19",
        "body_ratio_lag_20", "body_ratio_lag_21", "body_ratio_lag_22", "body_ratio_lag_23", "body_ratio_lag_24",
        "body_ratio_lag_25", "body_ratio_lag_26", "body_ratio_lag_27", "body_ratio_lag_28", "body_ratio_lag_29",
        "body_ratio_lag_30", "body_ratio_lag_31", "body_ratio_lag_32", "body_ratio_lag_33", "body_ratio_lag_34",
        "body_ratio_lag_35", "body_ratio_lag_36", "body_ratio_lag_37", "body_ratio_lag_38", "body_ratio_lag_39",
        "body_ratio_lag_40", "body_ratio_lag_41", "body_ratio_lag_42", "body_ratio_lag_43", "body_ratio_lag_44",
        "body_ratio_lag_45", "body_ratio_lag_46", "body_ratio_lag_47", "body_ratio_lag_48", "body_ratio_lag_49",
        "body_ratio_lag_50", "body_ratio_lag_51", "body_ratio_lag_52", "body_ratio_lag_53", "body_ratio_lag_54",
        "body_ratio_lag_55", "body_ratio_lag_56", "body_ratio_lag_57", "body_ratio_lag_58", "body_ratio_lag_59",
        "range_lag_0", "range_lag_1", "range_lag_2", "range_lag_3", "range_lag_4",
        "range_lag_5", "range_lag_6", "range_lag_7", "range_lag_8", "range_lag_9",
        "range_lag_10", "range_lag_11", "range_lag_12", "range_lag_13", "range_lag_14",
        "range_lag_15", "range_lag_16", "range_lag_17", "range_lag_18", "range_lag_19",
        "range_lag_20", "range_lag_21", "range_lag_22", "range_lag_23", "range_lag_24",
        "range_lag_25", "range_lag_26", "range_lag_27", "range_lag_28", "range_lag_29",
        "range_lag_30", "range_lag_31", "range_lag_32", "range_lag_33", "range_lag_34",
        "range_lag_35", "range_lag_36", "range_lag_37", "range_lag_38", "range_lag_39",
        "range_lag_40", "range_lag_41", "range_lag_42", "range_lag_43", "range_lag_44",
        "range_lag_45", "range_lag_46", "range_lag_47", "range_lag_48", "range_lag_49",
        "range_lag_50", "range_lag_51", "range_lag_52", "range_lag_53", "range_lag_54",
        "range_lag_55", "range_lag_56", "range_lag_57", "range_lag_58", "range_lag_59",
        "close_pos_lag_0", "close_pos_lag_1", "close_pos_lag_2", "close_pos_lag_3", "close_pos_lag_4",
        "close_pos_lag_5", "close_pos_lag_6", "close_pos_lag_7", "close_pos_lag_8", "close_pos_lag_9",
        "close_pos_lag_10", "close_pos_lag_11", "close_pos_lag_12", "close_pos_lag_13", "close_pos_lag_14",
        "close_pos_lag_15", "close_pos_lag_16", "close_pos_lag_17", "close_pos_lag_18", "close_pos_lag_19",
        "close_pos_lag_20", "close_pos_lag_21", "close_pos_lag_22", "close_pos_lag_23", "close_pos_lag_24",
        "close_pos_lag_25", "close_pos_lag_26", "close_pos_lag_27", "close_pos_lag_28", "close_pos_lag_29",
        "close_pos_lag_30", "close_pos_lag_31", "close_pos_lag_32", "close_pos_lag_33", "close_pos_lag_34",
        "close_pos_lag_35", "close_pos_lag_36", "close_pos_lag_37", "close_pos_lag_38", "close_pos_lag_39",
        "close_pos_lag_40", "close_pos_lag_41", "close_pos_lag_42", "close_pos_lag_43", "close_pos_lag_44",
        "close_pos_lag_45", "close_pos_lag_46", "close_pos_lag_47", "close_pos_lag_48", "close_pos_lag_49",
        "close_pos_lag_50", "close_pos_lag_51", "close_pos_lag_52", "close_pos_lag_53", "close_pos_lag_54",
        "close_pos_lag_55", "close_pos_lag_56", "close_pos_lag_57", "close_pos_lag_58", "close_pos_lag_59",
        "wick_ratio_lag_0", "wick_ratio_lag_1", "wick_ratio_lag_2", "wick_ratio_lag_3", "wick_ratio_lag_4",
        "wick_ratio_lag_5", "wick_ratio_lag_6", "wick_ratio_lag_7", "wick_ratio_lag_8", "wick_ratio_lag_9",
        "wick_ratio_lag_10", "wick_ratio_lag_11", "wick_ratio_lag_12", "wick_ratio_lag_13", "wick_ratio_lag_14",
        "wick_ratio_lag_15", "wick_ratio_lag_16", "wick_ratio_lag_17", "wick_ratio_lag_18", "wick_ratio_lag_19",
        "wick_ratio_lag_20", "wick_ratio_lag_21", "wick_ratio_lag_22", "wick_ratio_lag_23", "wick_ratio_lag_24",
        "wick_ratio_lag_25", "wick_ratio_lag_26", "wick_ratio_lag_27", "wick_ratio_lag_28", "wick_ratio_lag_29",
        "wick_ratio_lag_30", "wick_ratio_lag_31", "wick_ratio_lag_32", "wick_ratio_lag_33", "wick_ratio_lag_34",
        "wick_ratio_lag_35", "wick_ratio_lag_36", "wick_ratio_lag_37", "wick_ratio_lag_38", "wick_ratio_lag_39",
        "wick_ratio_lag_40", "wick_ratio_lag_41", "wick_ratio_lag_42", "wick_ratio_lag_43", "wick_ratio_lag_44",
        "wick_ratio_lag_45", "wick_ratio_lag_46", "wick_ratio_lag_47", "wick_ratio_lag_48", "wick_ratio_lag_49",
        "wick_ratio_lag_50", "wick_ratio_lag_51", "wick_ratio_lag_52", "wick_ratio_lag_53", "wick_ratio_lag_54",
        "wick_ratio_lag_55", "wick_ratio_lag_56", "wick_ratio_lag_57", "wick_ratio_lag_58", "wick_ratio_lag_59",
        "liq_sweep_flag_lag_0", "liq_sweep_flag_lag_1", "liq_sweep_flag_lag_2", "liq_sweep_flag_lag_3", "liq_sweep_flag_lag_4",
        "liq_sweep_flag_lag_5", "liq_sweep_flag_lag_6", "liq_sweep_flag_lag_7", "liq_sweep_flag_lag_8", "liq_sweep_flag_lag_9",
        "liq_sweep_flag_lag_10", "liq_sweep_flag_lag_11", "liq_sweep_flag_lag_12", "liq_sweep_flag_lag_13", "liq_sweep_flag_lag_14",
        "liq_sweep_flag_lag_15", "liq_sweep_flag_lag_16", "liq_sweep_flag_lag_17", "liq_sweep_flag_lag_18", "liq_sweep_flag_lag_19",
        "liq_sweep_flag_lag_20", "liq_sweep_flag_lag_21", "liq_sweep_flag_lag_22", "liq_sweep_flag_lag_23", "liq_sweep_flag_lag_24",
        "liq_sweep_flag_lag_25", "liq_sweep_flag_lag_26", "liq_sweep_flag_lag_27", "liq_sweep_flag_lag_28", "liq_sweep_flag_lag_29",
        "liq_sweep_flag_lag_30", "liq_sweep_flag_lag_31", "liq_sweep_flag_lag_32", "liq_sweep_flag_lag_33", "liq_sweep_flag_lag_34",
        "liq_sweep_flag_lag_35", "liq_sweep_flag_lag_36", "liq_sweep_flag_lag_37", "liq_sweep_flag_lag_38", "liq_sweep_flag_lag_39",
        "liq_sweep_flag_lag_40", "liq_sweep_flag_lag_41", "liq_sweep_flag_lag_42", "liq_sweep_flag_lag_43", "liq_sweep_flag_lag_44",
        "liq_sweep_flag_lag_45", "liq_sweep_flag_lag_46", "liq_sweep_flag_lag_47", "liq_sweep_flag_lag_48", "liq_sweep_flag_lag_49",
        "liq_sweep_flag_lag_50", "liq_sweep_flag_lag_51", "liq_sweep_flag_lag_52", "liq_sweep_flag_lag_53", "liq_sweep_flag_lag_54",
        "liq_sweep_flag_lag_55", "liq_sweep_flag_lag_56", "liq_sweep_flag_lag_57", "liq_sweep_flag_lag_58", "liq_sweep_flag_lag_59",
        "liq_sweep_overshoot_lag_0", "liq_sweep_overshoot_lag_1", "liq_sweep_overshoot_lag_2", "liq_sweep_overshoot_lag_3", "liq_sweep_overshoot_lag_4",
        "liq_sweep_overshoot_lag_5", "liq_sweep_overshoot_lag_6", "liq_sweep_overshoot_lag_7", "liq_sweep_overshoot_lag_8", "liq_sweep_overshoot_lag_9",
        "liq_sweep_overshoot_lag_10", "liq_sweep_overshoot_lag_11", "liq_sweep_overshoot_lag_12", "liq_sweep_overshoot_lag_13", "liq_sweep_overshoot_lag_14",
        "liq_sweep_overshoot_lag_15", "liq_sweep_overshoot_lag_16", "liq_sweep_overshoot_lag_17", "liq_sweep_overshoot_lag_18", "liq_sweep_overshoot_lag_19",
        "liq_sweep_overshoot_lag_20", "liq_sweep_overshoot_lag_21", "liq_sweep_overshoot_lag_22", "liq_sweep_overshoot_lag_23", "liq_sweep_overshoot_lag_24",
        "liq_sweep_overshoot_lag_25", "liq_sweep_overshoot_lag_26", "liq_sweep_overshoot_lag_27", "liq_sweep_overshoot_lag_28", "liq_sweep_overshoot_lag_29",
        "liq_sweep_overshoot_lag_30", "liq_sweep_overshoot_lag_31", "liq_sweep_overshoot_lag_32", "liq_sweep_overshoot_lag_33", "liq_sweep_overshoot_lag_34",
        "liq_sweep_overshoot_lag_35", "liq_sweep_overshoot_lag_36", "liq_sweep_overshoot_lag_37", "liq_sweep_overshoot_lag_38", "liq_sweep_overshoot_lag_39",
        "liq_sweep_overshoot_lag_40", "liq_sweep_overshoot_lag_41", "liq_sweep_overshoot_lag_42", "liq_sweep_overshoot_lag_43", "liq_sweep_overshoot_lag_44",
        "liq_sweep_overshoot_lag_45", "liq_sweep_overshoot_lag_46", "liq_sweep_overshoot_lag_47", "liq_sweep_overshoot_lag_48", "liq_sweep_overshoot_lag_49",
        "liq_sweep_overshoot_lag_50", "liq_sweep_overshoot_lag_51", "liq_sweep_overshoot_lag_52", "liq_sweep_overshoot_lag_53", "liq_sweep_overshoot_lag_54",
        "liq_sweep_overshoot_lag_55", "liq_sweep_overshoot_lag_56", "liq_sweep_overshoot_lag_57", "liq_sweep_overshoot_lag_58", "liq_sweep_overshoot_lag_59",
        "liq_reject_strength_lag_0", "liq_reject_strength_lag_1", "liq_reject_strength_lag_2", "liq_reject_strength_lag_3", "liq_reject_strength_lag_4",
        "liq_reject_strength_lag_5", "liq_reject_strength_lag_6", "liq_reject_strength_lag_7", "liq_reject_strength_lag_8", "liq_reject_strength_lag_9",
        "liq_reject_strength_lag_10", "liq_reject_strength_lag_11", "liq_reject_strength_lag_12", "liq_reject_strength_lag_13", "liq_reject_strength_lag_14",
        "liq_reject_strength_lag_15", "liq_reject_strength_lag_16", "liq_reject_strength_lag_17", "liq_reject_strength_lag_18", "liq_reject_strength_lag_19",
        "liq_reject_strength_lag_20", "liq_reject_strength_lag_21", "liq_reject_strength_lag_22", "liq_reject_strength_lag_23", "liq_reject_strength_lag_24",
        "liq_reject_strength_lag_25", "liq_reject_strength_lag_26", "liq_reject_strength_lag_27", "liq_reject_strength_lag_28", "liq_reject_strength_lag_29",
        "liq_reject_strength_lag_30", "liq_reject_strength_lag_31", "liq_reject_strength_lag_32", "liq_reject_strength_lag_33", "liq_reject_strength_lag_34",
        "liq_reject_strength_lag_35", "liq_reject_strength_lag_36", "liq_reject_strength_lag_37", "liq_reject_strength_lag_38", "liq_reject_strength_lag_39",
        "liq_reject_strength_lag_40", "liq_reject_strength_lag_41", "liq_reject_strength_lag_42", "liq_reject_strength_lag_43", "liq_reject_strength_lag_44",
        "liq_reject_strength_lag_45", "liq_reject_strength_lag_46", "liq_reject_strength_lag_47", "liq_reject_strength_lag_48", "liq_reject_strength_lag_49",
        "liq_reject_strength_lag_50", "liq_reject_strength_lag_51", "liq_reject_strength_lag_52", "liq_reject_strength_lag_53", "liq_reject_strength_lag_54",
        "liq_reject_strength_lag_55", "liq_reject_strength_lag_56", "liq_reject_strength_lag_57", "liq_reject_strength_lag_58", "liq_reject_strength_lag_59",
        "range_over_atr_lag_0", "range_over_atr_lag_1", "range_over_atr_lag_2", "range_over_atr_lag_3", "range_over_atr_lag_4",
        "range_over_atr_lag_5", "range_over_atr_lag_6", "range_over_atr_lag_7", "range_over_atr_lag_8", "range_over_atr_lag_9",
        "range_over_atr_lag_10", "range_over_atr_lag_11", "range_over_atr_lag_12", "range_over_atr_lag_13", "range_over_atr_lag_14",
        "range_over_atr_lag_15", "range_over_atr_lag_16", "range_over_atr_lag_17", "range_over_atr_lag_18", "range_over_atr_lag_19",
        "range_over_atr_lag_20", "range_over_atr_lag_21", "range_over_atr_lag_22", "range_over_atr_lag_23", "range_over_atr_lag_24",
        "range_over_atr_lag_25", "range_over_atr_lag_26", "range_over_atr_lag_27", "range_over_atr_lag_28", "range_over_atr_lag_29",
        "range_over_atr_lag_30", "range_over_atr_lag_31", "range_over_atr_lag_32", "range_over_atr_lag_33", "range_over_atr_lag_34",
        "range_over_atr_lag_35", "range_over_atr_lag_36", "range_over_atr_lag_37", "range_over_atr_lag_38", "range_over_atr_lag_39",
        "range_over_atr_lag_40", "range_over_atr_lag_41", "range_over_atr_lag_42", "range_over_atr_lag_43", "range_over_atr_lag_44",
        "range_over_atr_lag_45", "range_over_atr_lag_46", "range_over_atr_lag_47", "range_over_atr_lag_48", "range_over_atr_lag_49",
        "range_over_atr_lag_50", "range_over_atr_lag_51", "range_over_atr_lag_52", "range_over_atr_lag_53", "range_over_atr_lag_54",
        "range_over_atr_lag_55", "range_over_atr_lag_56", "range_over_atr_lag_57", "range_over_atr_lag_58", "range_over_atr_lag_59",
        "upper_wick_over_atr_lag_0", "upper_wick_over_atr_lag_1", "upper_wick_over_atr_lag_2", "upper_wick_over_atr_lag_3", "upper_wick_over_atr_lag_4",
        "upper_wick_over_atr_lag_5", "upper_wick_over_atr_lag_6", "upper_wick_over_atr_lag_7", "upper_wick_over_atr_lag_8", "upper_wick_over_atr_lag_9",
        "upper_wick_over_atr_lag_10", "upper_wick_over_atr_lag_11", "upper_wick_over_atr_lag_12", "upper_wick_over_atr_lag_13", "upper_wick_over_atr_lag_14",
        "upper_wick_over_atr_lag_15", "upper_wick_over_atr_lag_16", "upper_wick_over_atr_lag_17", "upper_wick_over_atr_lag_18", "upper_wick_over_atr_lag_19",
        "upper_wick_over_atr_lag_20", "upper_wick_over_atr_lag_21", "upper_wick_over_atr_lag_22", "upper_wick_over_atr_lag_23", "upper_wick_over_atr_lag_24",
        "upper_wick_over_atr_lag_25", "upper_wick_over_atr_lag_26", "upper_wick_over_atr_lag_27", "upper_wick_over_atr_lag_28", "upper_wick_over_atr_lag_29",
        "upper_wick_over_atr_lag_30", "upper_wick_over_atr_lag_31", "upper_wick_over_atr_lag_32", "upper_wick_over_atr_lag_33", "upper_wick_over_atr_lag_34",
        "upper_wick_over_atr_lag_35", "upper_wick_over_atr_lag_36", "upper_wick_over_atr_lag_37", "upper_wick_over_atr_lag_38", "upper_wick_over_atr_lag_39",
        "upper_wick_over_atr_lag_40", "upper_wick_over_atr_lag_41", "upper_wick_over_atr_lag_42", "upper_wick_over_atr_lag_43", "upper_wick_over_atr_lag_44",
        "upper_wick_over_atr_lag_45", "upper_wick_over_atr_lag_46", "upper_wick_over_atr_lag_47", "upper_wick_over_atr_lag_48", "upper_wick_over_atr_lag_49",
        "upper_wick_over_atr_lag_50", "upper_wick_over_atr_lag_51", "upper_wick_over_atr_lag_52", "upper_wick_over_atr_lag_53", "upper_wick_over_atr_lag_54",
        "upper_wick_over_atr_lag_55", "upper_wick_over_atr_lag_56", "upper_wick_over_atr_lag_57", "upper_wick_over_atr_lag_58", "upper_wick_over_atr_lag_59",
        "signed_body_lag_0", "signed_body_lag_1", "signed_body_lag_2", "signed_body_lag_3", "signed_body_lag_4",
        "signed_body_lag_5", "signed_body_lag_6", "signed_body_lag_7", "signed_body_lag_8", "signed_body_lag_9",
        "signed_body_lag_10", "signed_body_lag_11", "signed_body_lag_12", "signed_body_lag_13", "signed_body_lag_14",
        "signed_body_lag_15", "signed_body_lag_16", "signed_body_lag_17", "signed_body_lag_18", "signed_body_lag_19",
        "signed_body_lag_20", "signed_body_lag_21", "signed_body_lag_22", "signed_body_lag_23", "signed_body_lag_24",
        "signed_body_lag_25", "signed_body_lag_26", "signed_body_lag_27", "signed_body_lag_28", "signed_body_lag_29",
        "signed_body_lag_30", "signed_body_lag_31", "signed_body_lag_32", "signed_body_lag_33", "signed_body_lag_34",
        "signed_body_lag_35", "signed_body_lag_36", "signed_body_lag_37", "signed_body_lag_38", "signed_body_lag_39",
        "signed_body_lag_40", "signed_body_lag_41", "signed_body_lag_42", "signed_body_lag_43", "signed_body_lag_44",
        "signed_body_lag_45", "signed_body_lag_46", "signed_body_lag_47", "signed_body_lag_48", "signed_body_lag_49",
        "signed_body_lag_50", "signed_body_lag_51", "signed_body_lag_52", "signed_body_lag_53", "signed_body_lag_54",
        "signed_body_lag_55", "signed_body_lag_56", "signed_body_lag_57", "signed_body_lag_58", "signed_body_lag_59",
        "climax_vr_lag_0", "climax_vr_lag_1", "climax_vr_lag_2", "climax_vr_lag_3", "climax_vr_lag_4",
        "climax_vr_lag_5", "climax_vr_lag_6", "climax_vr_lag_7", "climax_vr_lag_8", "climax_vr_lag_9",
        "climax_vr_lag_10", "climax_vr_lag_11", "climax_vr_lag_12", "climax_vr_lag_13", "climax_vr_lag_14",
        "climax_vr_lag_15", "climax_vr_lag_16", "climax_vr_lag_17", "climax_vr_lag_18", "climax_vr_lag_19",
        "climax_vr_lag_20", "climax_vr_lag_21", "climax_vr_lag_22", "climax_vr_lag_23", "climax_vr_lag_24",
        "climax_vr_lag_25", "climax_vr_lag_26", "climax_vr_lag_27", "climax_vr_lag_28", "climax_vr_lag_29",
        "climax_vr_lag_30", "climax_vr_lag_31", "climax_vr_lag_32", "climax_vr_lag_33", "climax_vr_lag_34",
        "climax_vr_lag_35", "climax_vr_lag_36", "climax_vr_lag_37", "climax_vr_lag_38", "climax_vr_lag_39",
        "climax_vr_lag_40", "climax_vr_lag_41", "climax_vr_lag_42", "climax_vr_lag_43", "climax_vr_lag_44",
        "climax_vr_lag_45", "climax_vr_lag_46", "climax_vr_lag_47", "climax_vr_lag_48", "climax_vr_lag_49",
        "climax_vr_lag_50", "climax_vr_lag_51", "climax_vr_lag_52", "climax_vr_lag_53", "climax_vr_lag_54",
        "climax_vr_lag_55", "climax_vr_lag_56", "climax_vr_lag_57", "climax_vr_lag_58", "climax_vr_lag_59",
        "ret_accel_lag_0", "ret_accel_lag_1", "ret_accel_lag_2", "ret_accel_lag_3", "ret_accel_lag_4",
        "ret_accel_lag_5", "ret_accel_lag_6", "ret_accel_lag_7", "ret_accel_lag_8", "ret_accel_lag_9",
        "ret_accel_lag_10", "ret_accel_lag_11", "ret_accel_lag_12", "ret_accel_lag_13", "ret_accel_lag_14",
        "ret_accel_lag_15", "ret_accel_lag_16", "ret_accel_lag_17", "ret_accel_lag_18", "ret_accel_lag_19",
        "ret_accel_lag_20", "ret_accel_lag_21", "ret_accel_lag_22", "ret_accel_lag_23", "ret_accel_lag_24",
        "ret_accel_lag_25", "ret_accel_lag_26", "ret_accel_lag_27", "ret_accel_lag_28", "ret_accel_lag_29",
        "ret_accel_lag_30", "ret_accel_lag_31", "ret_accel_lag_32", "ret_accel_lag_33", "ret_accel_lag_34",
        "ret_accel_lag_35", "ret_accel_lag_36", "ret_accel_lag_37", "ret_accel_lag_38", "ret_accel_lag_39",
        "ret_accel_lag_40", "ret_accel_lag_41", "ret_accel_lag_42", "ret_accel_lag_43", "ret_accel_lag_44",
        "ret_accel_lag_45", "ret_accel_lag_46", "ret_accel_lag_47", "ret_accel_lag_48", "ret_accel_lag_49",
        "ret_accel_lag_50", "ret_accel_lag_51", "ret_accel_lag_52", "ret_accel_lag_53", "ret_accel_lag_54",
        "ret_accel_lag_55", "ret_accel_lag_56", "ret_accel_lag_57", "ret_accel_lag_58", "ret_accel_lag_59",
        "rsi_14_max_60", "rsi_14_min_60", "rsi_14_mean_60", "rsi_14_std_60", "rsi_14_last_minus_max_60",
        "rsi_14_slope_5", "rsi_14_delta_1", "rsi_14_delta_3", "rsi_14_delta_5",
        "mfi_14_max_60", "mfi_14_min_60", "mfi_14_mean_60", "mfi_14_std_60", "mfi_14_last_minus_max_60",
        "mfi_14_slope_5", "mfi_14_delta_1", "mfi_14_delta_3", "mfi_14_delta_5",
        "macdh_12_26_9_max_60", "macdh_12_26_9_min_60", "macdh_12_26_9_mean_60", "macdh_12_26_9_std_60", "macdh_12_26_9_last_minus_max_60",
        "macdh_12_26_9_slope_5", "macdh_12_26_9_delta_1", "macdh_12_26_9_delta_3", "macdh_12_26_9_delta_5",
        "macd_line_max_60", "macd_line_min_60", "macd_line_mean_60", "macd_line_std_60", "macd_line_last_minus_max_60",
        "macd_line_slope_5", "macd_line_delta_1", "macd_line_delta_3", "macd_line_delta_5",
        "vol_ratio_max_60", "vol_ratio_min_60", "vol_ratio_mean_60", "vol_ratio_std_60", "vol_ratio_last_minus_max_60",
        "vol_ratio_slope_5", "vol_ratio_delta_1", "vol_ratio_delta_3", "vol_ratio_delta_5",
        "ret_1_max_60", "ret_1_min_60", "ret_1_mean_60", "ret_1_std_60", "ret_1_last_minus_max_60",
        "ret_1_slope_5", "ret_1_delta_1", "ret_1_delta_3", "ret_1_delta_5",
        "drawdown_max_60", "drawdown_min_60", "drawdown_mean_60", "drawdown_std_60", "drawdown_last_minus_max_60",
        "drawdown_slope_5", "drawdown_delta_1", "drawdown_delta_3", "drawdown_delta_5",
        "rsi_14_minus_corridor", "mfi_14_minus_corridor", "macdh_12_26_9_minus_corridor",
        "runup", "runup_met", "vol_spike_cond", "vol_spike_recent",
        "rsi_hot", "mfi_hot", "osc_hot_recent", "macd_pos_recent",
        "pump_ctx", "near_peak", "blowoff_exhaustion",
        "osc_extreme", "predump_mask", "vol_fade", "rsi_fade", "macd_fade",
        "predump_peak", "strong_cond", "pump_score",
        "dist_to_pdh", "dist_to_pwh", "dist_to_eqh",
        "touched_pdh", "touched_pwh", "sweep_pdh", "sweep_pwh", "sweep_eqh",
        "overshoot_pdh", "overshoot_pwh", "overshoot_eqh",
        "eqh_strength", "eqh_age_bars",
        "liq_level_type_pwh", "liq_level_type_pdh", "liq_level_type_eqh",
        "liq_level_dist",
        "cum_ret_5", "cum_ret_10", "cum_ret_60",
        "count_red_last_5", "max_upper_wick_last_5",
        "vol_ratio_max_10", "vol_ratio_slope_5", "volume_fade",
        "atr_norm", "bb_z", "bb_width", "vwap_dev",
        "dollar_vol_prev",
    ]
}

META_COLUMNS = [
    "symbol", "open_time", "event_id", "offset", "y",
    "pump_la_type", "runup_pct", "target", "timeframe", "window_bars", "warmup_bars"
]


def get_feature_columns(feature_set: str) -> list:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}. Available: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[feature_set].copy()


_worker_builder_cache = {}


def _process_symbol_worker(args):
    ch_dsn, symbol, events_data, window_bars, warmup_bars, feature_set, params_dict = args

    cache_key = (ch_dsn, window_bars, warmup_bars, feature_set, tuple(sorted(params_dict.items())) if params_dict else None)

    if cache_key not in _worker_builder_cache:
        _worker_builder_cache[cache_key] = PumpLongFeatureBuilder(
            ch_dsn=ch_dsn,
            window_bars=window_bars,
            warmup_bars=warmup_bars,
            feature_set=feature_set,
            params=PumpParams(**params_dict) if params_dict else None
        )

    builder = _worker_builder_cache[cache_key]

    events = pd.DataFrame(events_data)
    events['open_time'] = pd.to_datetime(events['open_time'])

    return builder._process_symbol(symbol, events)


class PumpLongFeatureBuilder:
    def __init__(
            self,
            ch_dsn: str = None,
            window_bars: int = 60,
            warmup_bars: int = 150,
            feature_set: str = "extended",
            params: PumpParams = None
    ):
        self.ch_dsn = ch_dsn
        self.loader = DataLoader(ch_dsn) if ch_dsn else None
        self.window_bars = window_bars
        self.warmup_bars = warmup_bars
        self.feature_set = feature_set
        self.params = params or DEFAULT_PUMP_PARAMS
        self.vol_ratio_period = 50
        self.vwap_period = 30
        self.corridor_window = self.params.corridor_window
        self.corridor_quantile = self.params.corridor_quantile

    def build(self, labels_df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
        labels_df = labels_df[labels_df['pump_la_type'].isin(['A', 'B'])].copy()

        if 'event_open_time' in labels_df.columns:
            labels_df['open_time'] = pd.to_datetime(labels_df['event_open_time'], utc=True).dt.tz_localize(None)
        elif 'timestamp' in labels_df.columns:
            labels_df['open_time'] = pd.to_datetime(labels_df['timestamp'], utc=True).dt.tz_localize(None)

        labels_df = labels_df.sort_values(['symbol', 'open_time'])

        grouped = labels_df.groupby('symbol', sort=False)
        symbols = list(grouped.groups.keys())

        if len(symbols) <= 2 or max_workers <= 1:
            all_rows = []
            for symbol, group in grouped:
                rows = self._process_symbol(symbol, group)
                all_rows.extend(rows)
        else:
            params_dict = {
                'runup_window': self.params.runup_window,
                'runup_threshold': self.params.runup_threshold,
                'context_window': self.params.context_window,
                'peak_window': self.params.peak_window,
                'peak_tol': self.params.peak_tol,
                'volume_median_window': self.params.volume_median_window,
                'vol_ratio_spike': self.params.vol_ratio_spike,
                'vol_fade_ratio': self.params.vol_fade_ratio,
                'corridor_window': self.params.corridor_window,
                'corridor_quantile': self.params.corridor_quantile,
                'rsi_hot': self.params.rsi_hot,
                'mfi_hot': self.params.mfi_hot,
                'rsi_extreme': self.params.rsi_extreme,
                'mfi_extreme': self.params.mfi_extreme,
                'rsi_fade_ratio': self.params.rsi_fade_ratio,
                'macd_fade_ratio': self.params.macd_fade_ratio,
                'wick_high': self.params.wick_high,
                'wick_low': self.params.wick_low,
                'close_pos_high': self.params.close_pos_high,
                'close_pos_low': self.params.close_pos_low,
                'wick_blowoff': self.params.wick_blowoff,
                'body_blowoff': self.params.body_blowoff,
                'cooldown_bars': self.params.cooldown_bars,
                'liquidity_window_bars': self.params.liquidity_window_bars,
                'eqh_min_touches': self.params.eqh_min_touches,
                'eqh_base_tol': self.params.eqh_base_tol,
                'eqh_atr_factor': self.params.eqh_atr_factor,
            }

            tasks = []
            for symbol, group in grouped:
                events_data = group[['open_time', 'pump_la_type', 'runup_pct']].to_dict('records')
                tasks.append((
                    self.ch_dsn,
                    symbol,
                    events_data,
                    self.window_bars,
                    self.warmup_bars,
                    self.feature_set,
                    params_dict
                ))

            all_rows = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for rows in executor.map(_process_symbol_worker, tasks):
                    all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_rows)
        return result_df

    def build_one_for_inference(
            self,
            df_candles: pd.DataFrame,
            symbol: str,
            decision_open_time: pd.Timestamp
    ) -> dict:
        df = df_candles

        expected_bucket_start = decision_open_time - timedelta(minutes=15)

        events = pd.DataFrame([{
            'open_time': expected_bucket_start,
            'pump_la_type': 'A',
            'runup_pct': 0
        }])

        df = self._calculate_base_indicators(df)
        df = self._calculate_pump_detector_features(df)
        df = self._calculate_liquidity_features(df, events)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        df.loc[decision_open_time] = np.nan

        df = self._apply_decision_shift(df)

        events_for_extract = pd.DataFrame([{
            'open_time': decision_open_time,
            'pump_la_type': 'A',
            'runup_pct': 0
        }])

        rows = self._extract_features_vectorized(df, symbol, events_for_extract)

        if not rows:
            return {}

        return rows[0]

    def build_many_for_inference(
            self,
            df_candles: pd.DataFrame,
            symbol: str,
            decision_open_times: list
    ) -> list:
        if not decision_open_times:
            return []

        df = df_candles.copy()

        decision_times_arr = np.array(decision_open_times, dtype='datetime64[ns]')
        liq_times_arr = decision_times_arr - np.timedelta64(15, 'm')

        events_liq = pd.DataFrame({
            'open_time': liq_times_arr,
            'pump_la_type': 'A',
            'runup_pct': 0
        })

        df = self._calculate_base_indicators(df)
        df = self._calculate_pump_detector_features(df)
        df = self._calculate_liquidity_features(df, events_liq)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        existing_idx = df.index
        missing_times = [dt for dt in decision_open_times if dt not in existing_idx]

        if missing_times:
            combined_idx = existing_idx.union(pd.DatetimeIndex(missing_times))
            df = df.reindex(combined_idx)

        df = self._apply_decision_shift(df)

        events_extract = pd.DataFrame({
            'open_time': decision_times_arr,
            'pump_la_type': 'A',
            'runup_pct': 0
        })

        rows = self._extract_features_vectorized(df, symbol, events_extract)

        return rows

    def _process_symbol(self, symbol: str, events: pd.DataFrame) -> list:
        t_min = events['open_time'].min()
        t_max = events['open_time'].max()

        buffer_bars = self.warmup_bars + self.window_bars + self.params.liquidity_window_bars + 21
        start_bucket = t_min - pd.Timedelta(minutes=buffer_bars * 15)
        end_bucket = t_max

        df = self.loader.load_candles_range(symbol, start_bucket, end_bucket)

        if df.empty:
            return []

        df = self._calculate_base_indicators(df)
        df = self._calculate_pump_detector_features(df)
        df = self._calculate_liquidity_features(df, events)

        if self.feature_set == "extended":
            df = self._calculate_extended_indicators(df)

        df = self._apply_decision_shift(df)

        self._validate_columns(df)

        return self._extract_features_vectorized(df, symbol, events)

    def _validate_columns(self, df: pd.DataFrame):
        required = ['rsi_14', 'mfi_14', 'macdh_12_26_9']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required indicator columns: {missing}")

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ret_1'] = df['close'] / df['close'].shift(1) - 1

        candle_range = df['high'] - df['low']
        df['range'] = candle_range / df['close']

        max_oc = np.maximum(df['open'], df['close'])
        min_oc = np.minimum(df['open'], df['close'])

        df['upper_wick_ratio'] = np.where(candle_range > 0, (df['high'] - max_oc) / candle_range, 0)
        df['lower_wick_ratio'] = np.where(candle_range > 0, (min_oc - df['low']) / candle_range, 0)
        df['body_ratio'] = np.where(candle_range > 0, np.abs(df['close'] - df['open']) / candle_range, 0)

        df['log_volume'] = np.log(df['volume'].replace(0, np.nan))

        vol_median = df['volume'].rolling(window=self.vol_ratio_period).median()
        df['vol_ratio'] = df['volume'] / vol_median

        df.ta.rsi(length=14, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)

        df = df.rename(columns={
            'RSI_14': 'rsi_14',
            'MFI_14': 'mfi_14',
            'MACDh_12_26_9': 'macdh_12_26_9',
            'MACD_12_26_9': 'macd_line',
            'MACDs_12_26_9': 'macd_signal'
        })

        atr_col = [c for c in df.columns if 'ATR' in c and '14' in c]
        if atr_col:
            df = df.rename(columns={atr_col[0]: 'atr_14'})

        rolling_max_close = df['close'].rolling(window=self.window_bars).max()
        df['drawdown'] = (df['close'] - rolling_max_close) / rolling_max_close

        df['rsi_corridor'] = df['rsi_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)
        df['mfi_corridor'] = df['mfi_14'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)
        df['macdh_corridor'] = df['macdh_12_26_9'].rolling(window=self.corridor_window).quantile(self.corridor_quantile)

        df['range_over_atr'] = (df['high'] - df['low']) / (df['atr_14'] + 1e-9)
        df['upper_wick_over_atr'] = (df['high'] - max_oc) / (df['atr_14'] + 1e-9)
        df['signed_body'] = (df['close'] - df['open']) / (df['close'] + 1e-9)
        df['ret_accel'] = df['ret_1'] - df['ret_1'].shift(1)

        return df

    def _calculate_pump_detector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params

        df['vol_median_pd'] = df['volume'].rolling(window=p.volume_median_window).median()
        df['vol_ratio_pd'] = df['volume'] / df['vol_median_pd']

        df['vol_ratio_max'] = df['vol_ratio_pd'].rolling(window=p.peak_window).max()
        df['rsi_max'] = df['rsi_14'].rolling(window=p.peak_window).max()
        df['macdh_max'] = df['macdh_12_26_9'].rolling(window=p.peak_window).max()
        df['high_max'] = df['high'].rolling(window=p.peak_window).max()

        min_price = df[['low', 'open']].min(axis=1)
        local_min = min_price.rolling(window=p.runup_window).min()
        df['runup'] = (df['high'] / local_min) - 1
        df['runup_met'] = ((local_min > 0) & (df['runup'] >= p.runup_threshold)).astype(int)

        df['vol_spike_cond'] = (df['vol_ratio_pd'] >= p.vol_ratio_spike).astype(int)
        df['vol_spike_recent'] = (df['vol_spike_cond'].rolling(window=p.context_window).sum() > 0).astype(int)

        rsi = df['rsi_14']
        mfi = df['mfi_14']
        macdh = df['macdh_12_26_9']

        rsi_corridor_pd = df['rsi_corridor']
        mfi_corridor_pd = df['mfi_corridor']

        df['rsi_hot'] = (
                    rsi.notna() & rsi_corridor_pd.notna() & (rsi >= np.maximum(p.rsi_hot, rsi_corridor_pd))).astype(int)
        df['mfi_hot'] = (
                    mfi.notna() & mfi_corridor_pd.notna() & (mfi >= np.maximum(p.mfi_hot, mfi_corridor_pd))).astype(int)
        df['osc_hot_recent'] = ((df['rsi_hot'] | df['mfi_hot']).rolling(window=p.context_window).sum() > 0).astype(int)

        df['macd_pos_recent'] = ((macdh.notna() & (macdh > 0)).rolling(window=p.context_window).sum() > 0).astype(int)

        df['pump_ctx'] = (
                    df['runup_met'] & df['vol_spike_recent'] & df['osc_hot_recent'] & df['macd_pos_recent']).astype(int)

        df['near_peak'] = (df['high_max'].notna() & (df['high_max'] > 0) & (
                    df['high'] >= df['high_max'] * (1 - p.peak_tol))).astype(int)

        candle_range = df['high'] - df['low']
        range_pos = candle_range > 0

        df['close_pos'] = np.where(range_pos, (df['close'] - df['low']) / candle_range, 0)

        max_oc = np.maximum(df['open'], df['close'])
        df['wick_ratio'] = np.where(range_pos, (df['high'] - max_oc) / candle_range, 0)

        body_size = np.abs(df['close'] - df['open'])
        df['body_ratio_pd'] = np.where(range_pos, body_size / candle_range, 0)

        bearish = df['close'] < df['open']

        df['blowoff_exhaustion'] = (
                (df['close_pos'] <= p.close_pos_low) |
                (bearish & (df['close_pos'] <= 0.45)) |
                ((df['wick_ratio'] >= p.wick_blowoff) & (df['body_ratio_pd'] <= p.body_blowoff))
        ).astype(int)

        df['osc_extreme'] = (rsi.notna() & mfi.notna() & (rsi >= p.rsi_extreme) & (mfi >= p.mfi_extreme)).astype(int)
        df['predump_mask'] = (df['osc_extreme'] & (df['close_pos'] >= p.close_pos_high)).astype(int)

        df['vol_fade'] = (df['vol_ratio_max'].notna() & (df['vol_ratio_max'] > 0) & (
                    df['vol_ratio_pd'] <= df['vol_ratio_max'] * p.vol_fade_ratio)).astype(int)
        df['rsi_fade'] = (
                    df['rsi_max'].notna() & (df['rsi_max'] > 0) & (rsi <= df['rsi_max'] * p.rsi_fade_ratio)).astype(int)
        df['macd_fade'] = (df['macdh_max'].notna() & macdh.notna() & (df['macdh_max'] > 0) & (
                    macdh <= df['macdh_max'] * p.macd_fade_ratio)).astype(int)

        wick_high_mask = df['wick_ratio'] >= p.wick_high
        wick_low_mask = (df['wick_ratio'] >= p.wick_low) & (~wick_high_mask)

        df['predump_peak'] = (
                df['predump_mask'].astype(bool) &
                (
                        (wick_high_mask & df['vol_fade'].astype(bool)) |
                        (wick_low_mask & df['vol_fade'].astype(bool) & (
                                    df['rsi_fade'].astype(bool) | df['macd_fade'].astype(bool)))
                )
        ).astype(int)

        df['strong_cond'] = (df['pump_ctx'].astype(bool) & df['near_peak'].astype(bool) & (
                    df['blowoff_exhaustion'].astype(bool) | df['predump_peak'].astype(bool))).astype(int)

        df['pump_score'] = df['pump_ctx'] + df['near_peak'] + (df['blowoff_exhaustion'] | df['predump_peak']).astype(
            int)

        df['climax_vr'] = df['vol_ratio_pd'] * df['range']

        return df

    def _calculate_liquidity_features(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)

        df['pdh'] = np.nan
        df['pwh'] = np.nan

        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'date'):
            idx_dt = pd.to_datetime(df.index)
            dates = idx_dt.date
            df['_date'] = dates

            daily_max = df.groupby('_date')['high'].transform('max')
            daily_max_df = df.groupby('_date')['high'].max()
            daily_max_shifted = daily_max_df.shift(1)
            df['pdh'] = df['_date'].map(daily_max_shifted).values

            weeks = idx_dt.isocalendar()
            year_week_key = weeks.year * 100 + weeks.week
            df['_year_week'] = year_week_key.values

            weekly_max = df.groupby('_year_week')['high'].max()
            weekly_max_shifted = weekly_max.shift(1)
            df['pwh'] = df['_year_week'].map(weekly_max_shifted).values

            df.drop(columns=['_date', '_year_week'], inplace=True)

        df['dist_to_pdh'] = np.where(df['pdh'].notna() & (df['pdh'] > 0), (df['pdh'] - df['close']) / df['close'],
                                     np.nan)
        df['dist_to_pwh'] = np.where(df['pwh'].notna() & (df['pwh'] > 0), (df['pwh'] - df['close']) / df['close'],
                                     np.nan)

        df['touched_pdh'] = ((df['high'] >= df['pdh'] * 0.999) & df['pdh'].notna()).astype(int)
        df['touched_pwh'] = ((df['high'] >= df['pwh'] * 0.999) & df['pwh'].notna()).astype(int)

        df['sweep_pdh'] = (
                    (df['high'] > df['pdh'] * 1.001) & (df['close'] < df['pdh'] * 0.999) & df['pdh'].notna()).astype(
            int)
        df['sweep_pwh'] = (
                    (df['high'] > df['pwh'] * 1.001) & (df['close'] < df['pwh'] * 0.999) & df['pwh'].notna()).astype(
            int)

        df['overshoot_pdh'] = np.where(df['sweep_pdh'] == 1, (df['high'] - df['pdh']) / df['pdh'], 0)
        df['overshoot_pwh'] = np.where(df['sweep_pwh'] == 1, (df['high'] - df['pwh']) / df['pwh'], 0)

        liq_window = p.liquidity_window_bars
        eqh_tol_base = p.eqh_base_tol
        eqh_atr_factor = p.eqh_atr_factor
        min_touches = p.eqh_min_touches

        high_arr = df['high'].values
        atr_arr = df['atr_14'].values if 'atr_14' in df.columns else np.full(n, np.nan)
        close_arr = df['close'].values

        event_times = events['open_time'].values
        event_positions = df.index.get_indexer(pd.DatetimeIndex(event_times))

        needed_indices = set()
        required_history = self.warmup_bars + self.window_bars
        for pos in event_positions:
            if pos >= 0 and pos >= required_history:
                for offset in range(self.window_bars + 1):
                    idx = pos - offset
                    if idx >= liq_window:
                        needed_indices.add(idx)

        eqh_level_arr = np.full(n, np.nan)
        eqh_strength_arr = np.zeros(n, dtype=int)
        eqh_age_arr = np.full(n, np.nan)

        for i in sorted(needed_indices):
            if i < liq_window or i >= n:
                continue

            window_highs = high_arr[i - liq_window:i]
            original_indices = np.arange(liq_window)
            current_atr = atr_arr[i - 1] if i > 0 and not np.isnan(atr_arr[i - 1]) else 0
            current_close = close_arr[i - 1] if i > 0 else close_arr[i]
            tol = max(eqh_tol_base, eqh_atr_factor * current_atr / current_close if current_close > 0 else eqh_tol_base)

            sorted_order = np.argsort(window_highs)[::-1]
            sorted_highs = window_highs[sorted_order]
            sorted_orig_idx = original_indices[sorted_order]

            clusters = []
            cluster_start = 0

            while cluster_start < liq_window:
                level = sorted_highs[cluster_start]
                if level <= 0:
                    cluster_start += 1
                    continue

                threshold = level * (1 - tol)
                cluster_end = cluster_start + 1
                while cluster_end < liq_window and sorted_highs[cluster_end] >= threshold:
                    cluster_end += 1

                cluster_size = cluster_end - cluster_start
                if cluster_size >= min_touches:
                    cluster_highs = sorted_highs[cluster_start:cluster_end]
                    cluster_orig_indices = sorted_orig_idx[cluster_start:cluster_end]
                    avg_level = np.mean(cluster_highs)
                    last_touch_age = liq_window - np.max(cluster_orig_indices) - 1
                    clusters.append((avg_level, cluster_size, last_touch_age))

                cluster_start = cluster_end

            if clusters:
                clusters.sort(key=lambda x: (-x[1], -x[0]))
                best = clusters[0]
                eqh_level_arr[i] = best[0]
                eqh_strength_arr[i] = best[1]
                eqh_age_arr[i] = best[2]

        df['eqh_level'] = eqh_level_arr
        df['eqh_strength'] = eqh_strength_arr
        df['eqh_age_bars'] = eqh_age_arr

        df['dist_to_eqh'] = np.where(df['eqh_level'].notna() & (df['eqh_level'] > 0),
                                     (df['eqh_level'] - df['close']) / df['close'], np.nan)
        df['sweep_eqh'] = ((df['high'] > df['eqh_level'] * 1.001) & (df['close'] < df['eqh_level'] * 0.999) & df[
            'eqh_level'].notna()).astype(int)
        df['overshoot_eqh'] = np.where(df['sweep_eqh'] == 1, (df['high'] - df['eqh_level']) / df['eqh_level'], 0)

        sweep_pwh = df['sweep_pwh'].values
        sweep_pdh = df['sweep_pdh'].values
        sweep_eqh = df['sweep_eqh'].values
        dist_pwh = df['dist_to_pwh'].values
        dist_pdh = df['dist_to_pdh'].values
        dist_eqh = df['dist_to_eqh'].values
        overshoot_pwh = df['overshoot_pwh'].values
        overshoot_pdh = df['overshoot_pdh'].values
        overshoot_eqh = df['overshoot_eqh'].values

        liq_type_pwh = np.zeros(n, dtype=int)
        liq_type_pdh = np.zeros(n, dtype=int)
        liq_type_eqh = np.zeros(n, dtype=int)
        liq_dist = np.full(n, np.nan)
        liq_sweep = np.zeros(n, dtype=int)
        liq_overshoot = np.zeros(n, dtype=float)

        mask_pwh = sweep_pwh == 1
        liq_type_pwh[mask_pwh] = 1
        liq_dist[mask_pwh] = dist_pwh[mask_pwh]
        liq_sweep[mask_pwh] = 1
        liq_overshoot[mask_pwh] = overshoot_pwh[mask_pwh]

        mask_pdh = (sweep_pdh == 1) & ~mask_pwh
        liq_type_pdh[mask_pdh] = 1
        liq_dist[mask_pdh] = dist_pdh[mask_pdh]
        liq_sweep[mask_pdh] = 1
        liq_overshoot[mask_pdh] = overshoot_pdh[mask_pdh]

        mask_eqh = (sweep_eqh == 1) & ~mask_pwh & ~mask_pdh
        liq_type_eqh[mask_eqh] = 1
        liq_dist[mask_eqh] = dist_eqh[mask_eqh]
        liq_sweep[mask_eqh] = 1
        liq_overshoot[mask_eqh] = overshoot_eqh[mask_eqh]

        mask_no_sweep = ~mask_pwh & ~mask_pdh & ~mask_eqh

        dist_pwh_clean = np.where((~np.isnan(dist_pwh)) & (dist_pwh > 0), dist_pwh, np.inf)
        dist_pdh_clean = np.where((~np.isnan(dist_pdh)) & (dist_pdh > 0), dist_pdh, np.inf)
        dist_eqh_clean = np.where((~np.isnan(dist_eqh)) & (dist_eqh > 0), dist_eqh, np.inf)

        dist_stack = np.stack([dist_pwh_clean, dist_pdh_clean, dist_eqh_clean], axis=1)
        min_idx = np.argmin(dist_stack, axis=1)
        min_dist = np.min(dist_stack, axis=1)

        has_candidate = mask_no_sweep & (min_dist < np.inf)

        liq_type_pwh[has_candidate & (min_idx == 0)] = 1
        liq_type_pdh[has_candidate & (min_idx == 1)] = 1
        liq_type_eqh[has_candidate & (min_idx == 2)] = 1
        liq_dist[has_candidate] = min_dist[has_candidate]

        df['liq_level_type_pwh'] = liq_type_pwh
        df['liq_level_type_pdh'] = liq_type_pdh
        df['liq_level_type_eqh'] = liq_type_eqh
        df['liq_level_dist'] = liq_dist
        df['liq_sweep_flag'] = liq_sweep
        df['liq_sweep_overshoot'] = liq_overshoot

        close_pos = df['close_pos'].values
        wick_ratio = df['wick_ratio'].values
        df['liq_reject_strength'] = np.where(
            (close_pos <= 0.4) & (wick_ratio >= 0.25),
            (0.4 - close_pos) + (wick_ratio - 0.25),
            0
        )

        return df

    def _calculate_extended_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'atr_14' not in df.columns:
            df.ta.atr(length=14, append=True)
            atr_col = [c for c in df.columns if 'ATR' in c and '14' in c]
            if atr_col:
                df = df.rename(columns={atr_col[0]: 'atr_14'})

        df['atr_norm'] = df['atr_14'] / df['close']

        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_z'] = (df['close'] - sma_20) / std_20
        df['bb_width'] = std_20 / sma_20

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).rolling(window=self.vwap_period).sum()
        cumulative_vol = df['volume'].rolling(window=self.vwap_period).sum()
        vwap = cumulative_tp_vol / cumulative_vol
        df['vwap_dev'] = (df['close'] - vwap) / vwap

        df['dollar_vol_prev'] = df['close'] * df['volume']

        df.ta.obv(append=True)
        df = df.rename(columns={'OBV': 'obv'})

        return df

    def _apply_decision_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        shift_columns = [
            'ret_1', 'range', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'volume', 'log_volume', 'rsi_14', 'mfi_14', 'macdh_12_26_9', 'vol_ratio',
            'macd_line', 'macd_signal', 'drawdown',
            'rsi_corridor', 'mfi_corridor', 'macdh_corridor',
            'vol_median_pd', 'vol_ratio_pd', 'vol_ratio_max', 'rsi_max', 'macdh_max', 'high_max',
            'runup', 'runup_met', 'vol_spike_cond', 'vol_spike_recent',
            'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'macd_pos_recent',
            'pump_ctx', 'near_peak', 'close_pos', 'wick_ratio', 'body_ratio_pd',
            'blowoff_exhaustion', 'osc_extreme', 'predump_mask',
            'vol_fade', 'rsi_fade', 'macd_fade', 'predump_peak', 'strong_cond', 'pump_score',
            'pdh', 'pwh', 'dist_to_pdh', 'dist_to_pwh',
            'touched_pdh', 'touched_pwh', 'sweep_pdh', 'sweep_pwh',
            'overshoot_pdh', 'overshoot_pwh',
            'eqh_level', 'eqh_strength', 'eqh_age_bars', 'dist_to_eqh', 'sweep_eqh', 'overshoot_eqh',
            'liq_level_type_pwh', 'liq_level_type_pdh', 'liq_level_type_eqh',
            'liq_level_dist', 'liq_sweep_flag', 'liq_sweep_overshoot', 'liq_reject_strength',
            'range_over_atr', 'upper_wick_over_atr', 'signed_body', 'climax_vr', 'ret_accel'
        ]

        extended_columns = ['atr_14', 'atr_norm', 'bb_z', 'bb_width', 'vwap_dev', 'obv', 'dollar_vol_prev']

        cols_to_shift = [c for c in shift_columns if c in df.columns]
        if self.feature_set == "extended":
            cols_to_shift.extend([c for c in extended_columns if c in df.columns])

        if cols_to_shift:
            df[cols_to_shift] = df[cols_to_shift].shift(1)

        return df

    def _extract_features_vectorized(self, df: pd.DataFrame, symbol: str, events: pd.DataFrame) -> list:
        event_times = events['open_time'].values
        positions = df.index.get_indexer(pd.DatetimeIndex(event_times))

        required_history = self.warmup_bars + self.window_bars
        valid_mask = (positions >= 0) & (positions >= required_history)

        valid_positions = positions[valid_mask]
        valid_events = events.iloc[valid_mask]

        if len(valid_positions) == 0:
            return []

        lag_series = [
            'ret_1', 'vol_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio', 'range',
            'close_pos', 'wick_ratio',
            'liq_sweep_flag', 'liq_sweep_overshoot', 'liq_reject_strength',
            'range_over_atr', 'upper_wick_over_atr', 'signed_body', 'climax_vr', 'ret_accel'
        ]

        compact_series = ['rsi_14', 'mfi_14', 'macdh_12_26_9', 'macd_line', 'vol_ratio', 'ret_1', 'drawdown']

        pump_detector_features = [
            'runup', 'runup_met', 'vol_spike_cond', 'vol_spike_recent',
            'rsi_hot', 'mfi_hot', 'osc_hot_recent', 'macd_pos_recent',
            'pump_ctx', 'near_peak', 'blowoff_exhaustion',
            'osc_extreme', 'predump_mask', 'vol_fade', 'rsi_fade', 'macd_fade',
            'predump_peak', 'strong_cond', 'pump_score'
        ]

        liquidity_features = [
            'dist_to_pdh', 'dist_to_pwh', 'dist_to_eqh',
            'touched_pdh', 'touched_pwh', 'sweep_pdh', 'sweep_pwh', 'sweep_eqh',
            'overshoot_pdh', 'overshoot_pwh', 'overshoot_eqh',
            'eqh_strength', 'eqh_age_bars',
            'liq_level_type_pwh', 'liq_level_type_pdh', 'liq_level_type_eqh',
            'liq_level_dist'
        ]

        extended_features = ['atr_norm', 'bb_z', 'bb_width', 'vwap_dev', 'dollar_vol_prev']

        series_arrays = {}
        all_series = set(lag_series + compact_series + pump_detector_features + liquidity_features)
        if self.feature_set == "extended":
            all_series.update(extended_features)

        for s in all_series:
            if s in df.columns:
                series_arrays[s] = df[s].values

        close_arr = df['close'].values
        open_arr = df['open'].values
        volume_arr = df['volume'].values

        corridor_arrays = {}
        for name in ['rsi_corridor', 'mfi_corridor', 'macdh_corridor']:
            if name in df.columns:
                corridor_arrays[name] = df[name].values

        num_events = len(valid_positions)
        w = self.window_bars

        pos_arr = valid_positions
        idx_matrix = pos_arr[:, None] - np.arange(w)[None, :]

        event_open_times = df.index[valid_positions].values
        event_pump_types = valid_events['pump_la_type'].values
        event_targets = (valid_events['pump_la_type'].values == 'A').astype(int)
        event_runups = valid_events['runup_pct'].values

        result_data = {
            'symbol': np.full(num_events, symbol),
            'open_time': event_open_times,
            'pump_la_type': event_pump_types,
            'target': event_targets,
            'runup_pct': event_runups,
            'timeframe': np.full(num_events, '15m'),
            'window_bars': np.full(num_events, self.window_bars),
            'warmup_bars': np.full(num_events, self.warmup_bars)
        }

        for series_name in lag_series:
            if series_name not in series_arrays:
                continue
            arr = series_arrays[series_name]
            values_matrix = arr[idx_matrix]
            for lag in range(w):
                result_data[f'{series_name}_lag_{lag}'] = values_matrix[:, lag]

        for series_name in compact_series:
            if series_name not in series_arrays:
                continue
            arr = series_arrays[series_name]
            values_matrix = arr[idx_matrix]

            result_data[f'{series_name}_max_{w}'] = np.nanmax(values_matrix, axis=1)
            result_data[f'{series_name}_min_{w}'] = np.nanmin(values_matrix, axis=1)
            result_data[f'{series_name}_mean_{w}'] = np.nanmean(values_matrix, axis=1)
            result_data[f'{series_name}_std_{w}'] = np.nanstd(values_matrix, axis=1)
            result_data[f'{series_name}_last_minus_max_{w}'] = values_matrix[:, 0] - np.nanmax(values_matrix, axis=1)

            if w >= 5:
                result_data[f'{series_name}_slope_5'] = values_matrix[:, 0] - values_matrix[:, 4]
            else:
                result_data[f'{series_name}_slope_5'] = np.full(num_events, np.nan)

            result_data[f'{series_name}_delta_1'] = values_matrix[:, 0] - values_matrix[:, 1] if w >= 2 else np.full(num_events, np.nan)
            result_data[f'{series_name}_delta_3'] = values_matrix[:, 0] - values_matrix[:, 2] if w >= 3 else np.full(num_events, np.nan)
            result_data[f'{series_name}_delta_5'] = values_matrix[:, 0] - values_matrix[:, 4] if w >= 5 else np.full(num_events, np.nan)

        point_idx = pos_arr
        for corridor_name, base_name in [('rsi_corridor', 'rsi_14'), ('mfi_corridor', 'mfi_14'),
                                         ('macdh_corridor', 'macdh_12_26_9')]:
            if corridor_name in corridor_arrays and base_name in series_arrays:
                corridor_vals = corridor_arrays[corridor_name][point_idx]
                base_vals = series_arrays[base_name][point_idx]
                result_data[f'{base_name}_minus_corridor'] = np.where(
                    np.isnan(base_vals) | np.isnan(corridor_vals), np.nan, base_vals - corridor_vals
                )

        for feat_name in pump_detector_features:
            if feat_name in series_arrays:
                result_data[feat_name] = series_arrays[feat_name][point_idx]

        for feat_name in liquidity_features:
            if feat_name in series_arrays:
                result_data[feat_name] = series_arrays[feat_name][point_idx]

        if self.feature_set == "extended":
            for feat_name in extended_features:
                if feat_name in series_arrays:
                    result_data[feat_name] = series_arrays[feat_name][point_idx]

        cum_ret_5 = np.full(num_events, np.nan)
        mask_5 = pos_arr >= 6
        close_prev = close_arr[pos_arr[mask_5] - 1]
        close_5_ago = close_arr[pos_arr[mask_5] - 5]
        cum_ret_5[mask_5] = np.where(close_5_ago != 0, close_prev / close_5_ago - 1, np.nan)
        result_data['cum_ret_5'] = cum_ret_5

        cum_ret_10 = np.full(num_events, np.nan)
        mask_10 = pos_arr >= 11
        close_prev = close_arr[pos_arr[mask_10] - 1]
        close_10_ago = close_arr[pos_arr[mask_10] - 10]
        cum_ret_10[mask_10] = np.where(close_10_ago != 0, close_prev / close_10_ago - 1, np.nan)
        result_data['cum_ret_10'] = cum_ret_10

        cum_ret_w = np.full(num_events, np.nan)
        mask_w = pos_arr >= w + 1
        close_prev = close_arr[pos_arr[mask_w] - 1]
        close_w_ago = close_arr[pos_arr[mask_w] - w]
        cum_ret_w[mask_w] = np.where(close_w_ago != 0, close_prev / close_w_ago - 1, np.nan)
        result_data[f'cum_ret_{w}'] = cum_ret_w

        count_red = np.zeros(num_events, dtype=int)
        mask_red = pos_arr >= 6
        for i in np.where(mask_red)[0]:
            p = pos_arr[i]
            count_red[i] = np.sum(close_arr[p - 5:p] < open_arr[p - 5:p])
        result_data['count_red_last_5'] = count_red

        if 'upper_wick_ratio' in series_arrays and w >= 5:
            uw_matrix = series_arrays['upper_wick_ratio'][idx_matrix[:, :5]]
            result_data['max_upper_wick_last_5'] = np.nanmax(uw_matrix, axis=1)
        else:
            result_data['max_upper_wick_last_5'] = np.full(num_events, np.nan)

        if 'vol_ratio' in series_arrays:
            vr_matrix = series_arrays['vol_ratio'][idx_matrix]
            result_data['vol_ratio_max_10'] = np.nanmax(vr_matrix[:, :10], axis=1) if w >= 10 else np.full(num_events, np.nan)
            result_data['vol_ratio_slope_5'] = vr_matrix[:, 0] - vr_matrix[:, 4] if w >= 5 else np.full(num_events, np.nan)

        volume_fade = np.full(num_events, np.nan)
        mask_vf = pos_arr >= 10
        for i in np.where(mask_vf)[0]:
            p = pos_arr[i]
            vol_slice = volume_arr[p - 9:p + 1]
            max_vol = np.nanmax(vol_slice)
            if max_vol > 0:
                volume_fade[i] = volume_arr[p] / max_vol
        result_data['volume_fade'] = volume_fade

        result_df = pd.DataFrame(result_data)
        return result_df.to_dict('records')
