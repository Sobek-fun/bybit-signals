MODEL_REASON_CLASSES: tuple[str, ...] = (
    "good",
    "too_early",
    "too_late",
    "continuation_flat",
)

MODEL_REASON_CLASS_TO_ID: dict[str, int] = {
    "good": 0,
    "too_early": 1,
    "too_late": 2,
    "continuation_flat": 3,
}

TARGET_REASON_TO_MODEL_CLASS: dict[str, str | None] = {
    "good": "good",
    "too_early": "too_early",
    "too_late": "too_late",
    "continuation": "continuation_flat",
    "flat": "continuation_flat",
    "invalid_context": None,
}


def map_target_reason_to_model_class(target_reason: object) -> str | None:
    normalized_reason = str(target_reason).strip().lower()
    if normalized_reason not in TARGET_REASON_TO_MODEL_CLASS:
        raise ValueError(f"unknown target_reason for detector mapping: {target_reason!r}")
    return TARGET_REASON_TO_MODEL_CLASS[normalized_reason]
