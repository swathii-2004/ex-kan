from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import shap

PredictFn = Callable[[str], tuple[str, float]]
Method = Literal["lime", "shap"]

# Deletion: removing the top words must drop confidence by at least this much (or
# flip the prediction outright) to count as faithful — the words were necessary.
CONFIDENCE_DROP_THRESHOLD = 0.2

# Insertion: keeping only the top words must retain at least this much confidence in
# the original prediction to count as faithful — the words were sufficient on their own.
CONFIDENCE_RETENTION_THRESHOLD = 0.6

# Length-proportional top-N: a flat N touches a much larger fraction of short
# sentences than long ones, mechanically biasing deletion/insertion against longer
# text independent of explanation quality (see PROJECT_ROADMAP.md Phase 4).
TOP_N_FRACTION = 0.25
TOP_N_MIN = 3
TOP_N_MAX = 10


def adaptive_top_n(word_count: int) -> int:
    return max(TOP_N_MIN, min(TOP_N_MAX, round(word_count * TOP_N_FRACTION)))


@dataclass
class DeletionResult:
    text: str
    modified_text: str
    original_prediction: str
    original_confidence: float
    modified_prediction: str
    modified_confidence: float
    prediction_flipped: bool
    confidence_drop: float
    faithful: bool


@dataclass
class InsertionResult:
    text: str
    modified_text: str
    original_prediction: str
    original_confidence: float
    modified_prediction: str
    modified_confidence: float
    prediction_held: bool
    confidence_retained: float
    faithful: bool


def _remove_words(text: str, words: list[str]) -> str:
    drop = set(words)
    return " ".join(tok for tok in text.split() if tok not in drop)


def _keep_only_words(text: str, words: list[str]) -> str:
    keep = set(words)
    return " ".join(tok for tok in text.split() if tok in keep)


def _shap_segments(text: str, tokenizer: Any) -> list[str]:
    """The exact subword segmentation shap.maskers.Text uses (and that explain_shap's
    returned tokens come from), so SHAP's top_words align with these segments — not
    with whitespace words. Each segment carries its own trailing whitespace from the
    original text, so concatenating a subset reproduces correct spacing."""
    strings, _ = shap.maskers.Text(tokenizer).token_segments(text)
    return strings


def _remove_shap_tokens(text: str, words: list[str], tokenizer: Any) -> str:
    drop = {w.strip() for w in words}
    segments = _shap_segments(text, tokenizer)
    return "".join(s for s in segments if s.strip() not in drop).strip()


def _keep_only_shap_tokens(text: str, words: list[str], tokenizer: Any) -> str:
    keep = {w.strip() for w in words}
    segments = _shap_segments(text, tokenizer)
    return "".join(s for s in segments if s.strip() in keep).strip()


def deletion_test(
    text: str,
    top_words: list[str],
    model: PredictFn,
    method: Method = "lime",
    tokenizer: Any = None,
) -> DeletionResult:
    original_prediction, original_confidence = model(text)

    if method == "shap":
        if tokenizer is None:
            raise ValueError("method='shap' requires a tokenizer")
        modified_text = _remove_shap_tokens(text, top_words, tokenizer)
    else:
        modified_text = _remove_words(text, top_words)

    if not modified_text.strip():
        # Nothing left to predict on: the top words were the entire text, so removing
        # them is treated as maximal evidence of faithfulness rather than a model call
        # on an empty string.
        modified_prediction, modified_confidence = "undefined", 0.0
    else:
        modified_prediction, modified_confidence = model(modified_text)

    prediction_flipped = modified_prediction != original_prediction
    confidence_drop = original_confidence if prediction_flipped else original_confidence - modified_confidence
    faithful = prediction_flipped or confidence_drop >= CONFIDENCE_DROP_THRESHOLD

    return DeletionResult(
        text=text,
        modified_text=modified_text,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        modified_prediction=modified_prediction,
        modified_confidence=modified_confidence,
        prediction_flipped=prediction_flipped,
        confidence_drop=confidence_drop,
        faithful=faithful,
    )


def insertion_test(
    text: str,
    top_words: list[str],
    model: PredictFn,
    method: Method = "lime",
    tokenizer: Any = None,
) -> InsertionResult:
    original_prediction, original_confidence = model(text)

    if method == "shap":
        if tokenizer is None:
            raise ValueError("method='shap' requires a tokenizer")
        modified_text = _keep_only_shap_tokens(text, top_words, tokenizer)
    else:
        modified_text = _keep_only_words(text, top_words)

    if not modified_text.strip():
        modified_prediction, modified_confidence = "undefined", 0.0
    else:
        modified_prediction, modified_confidence = model(modified_text)

    prediction_held = modified_prediction == original_prediction
    confidence_retained = modified_confidence if prediction_held else 0.0
    faithful = prediction_held and confidence_retained >= CONFIDENCE_RETENTION_THRESHOLD

    return InsertionResult(
        text=text,
        modified_text=modified_text,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        modified_prediction=modified_prediction,
        modified_confidence=modified_confidence,
        prediction_held=prediction_held,
        confidence_retained=confidence_retained,
        faithful=faithful,
    )


def faithfulness_score(deletion_result: DeletionResult, insertion_result: InsertionResult) -> float:
    """Average of the deletion (necessity) and insertion (sufficiency) components,
    each clipped to [0, 1]. A faithful explanation scores high on both: the model
    falls apart without the top words, and holds up with only them."""
    deletion_component = min(max(deletion_result.confidence_drop, 0.0), 1.0)
    insertion_component = min(max(insertion_result.confidence_retained, 0.0), 1.0)
    return (deletion_component + insertion_component) / 2
