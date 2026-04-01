from pump_end_v2.detector.dataset import build_detector_dataset
from pump_end_v2.detector.model import (
    SequenceDetector,
    build_detector_feature_importance_table,
    build_detector_model,
    build_sequence_permutation_importance_table,
    fit_detector_model,
    predict_detector_scores,
    summarize_detector_oof_importance,
)
from pump_end_v2.detector.sequence_dataset import (
    DetectorSequenceStore,
    build_detector_sequence_store,
    extract_sequences_for_rows,
)
from pump_end_v2.detector.policy import (
    apply_episode_aware_detector_policy,
    build_detector_policy_metrics,
    compute_eval_window_days_from_policy_rows,
)
from pump_end_v2.detector.policy_search import (
    build_detector_policy_grid,
    build_detector_test_candidate_signal_ledger,
    build_detector_test_policy_rows,
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
from pump_end_v2.detector.target_metrics import (
    build_detector_rank_quality_report,
    build_detector_score_decile_report,
    build_detector_target_metrics,
)

__all__ = [
    "build_detector_dataset",
    "build_detector_sequence_store",
    "extract_sequences_for_rows",
    "DetectorSequenceStore",
    "assign_detector_dataset_splits",
    "summarize_detector_splits",
    "DetectorFold",
    "generate_detector_walkforward_folds",
    "filter_fold_rows",
    "build_detector_model",
    "SequenceDetector",
    "build_detector_feature_importance_table",
    "build_sequence_permutation_importance_table",
    "fit_detector_model",
    "predict_detector_scores",
    "summarize_detector_oof_importance",
    "apply_episode_aware_detector_policy",
    "build_detector_policy_metrics",
    "compute_eval_window_days_from_policy_rows",
    "build_detector_val_policy_rows",
    "build_detector_train_oof_policy_rows",
    "build_detector_policy_grid",
    "sweep_detector_policy",
    "select_detector_policy",
    "build_detector_val_candidate_signal_ledger",
    "build_detector_train_oof_candidate_signal_ledger",
    "build_detector_test_policy_rows",
    "build_detector_test_candidate_signal_ledger",
    "build_detector_target_metrics",
    "build_detector_score_decile_report",
    "build_detector_rank_quality_report",
]
