from pump_end_v2.detector.dataset import build_detector_dataset
from pump_end_v2.detector.model import build_detector_model, fit_detector_model, predict_detector_scores
from pump_end_v2.detector.oof import build_detector_train_oof_scores, fit_detector_on_train_and_score_val
from pump_end_v2.detector.policy import apply_episode_aware_detector_policy, build_detector_policy_metrics
from pump_end_v2.detector.policy_search import (
    build_detector_policy_grid,
    build_detector_train_oof_candidate_signal_ledger,
    build_detector_train_oof_policy_rows,
    build_detector_val_candidate_signal_ledger,
    build_detector_val_policy_rows,
    select_detector_policy,
    sweep_detector_policy,
)
from pump_end_v2.detector.splits import (
    DetectorFold,
    assign_detector_dataset_splits,
    filter_fold_rows,
    generate_detector_walkforward_folds,
    summarize_detector_splits,
)

__all__ = [
    "build_detector_dataset",
    "assign_detector_dataset_splits",
    "summarize_detector_splits",
    "DetectorFold",
    "generate_detector_walkforward_folds",
    "filter_fold_rows",
    "build_detector_model",
    "fit_detector_model",
    "predict_detector_scores",
    "build_detector_train_oof_scores",
    "fit_detector_on_train_and_score_val",
    "apply_episode_aware_detector_policy",
    "build_detector_policy_metrics",
    "build_detector_val_policy_rows",
    "build_detector_train_oof_policy_rows",
    "build_detector_policy_grid",
    "sweep_detector_policy",
    "select_detector_policy",
    "build_detector_val_candidate_signal_ledger",
    "build_detector_train_oof_candidate_signal_ledger",
]

