from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(__file__).resolve().parents[3] / "model" / "checkpoints" / "ex_kan_model"

# The checkpoint's config has no id2label mapping (generic LABEL_0/LABEL_1), since the
# training script didn't set one. Per PROJECT_ROADMAP.md Phase 2 ("Negative + Mixed
# feelings = distress"), label 1 is assumed to be the distress class. Verify against the
# training script if one becomes available.
LABELS = ["not-distress", "distress"]

_BATCH_SIZE = 32


@dataclass
class ExplanationResult:
    prediction: str
    confidence: float
    tokens: list[tuple[str, float]]


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def _predict_proba(texts: list[str]) -> np.ndarray:
    tokenizer, model = _load_model()
    probs = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = list(texts[i : i + _BATCH_SIZE])
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(probs, axis=0)


def _predict(text: str) -> tuple[str, float]:
    probs = _predict_proba([text])[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])


def _rank_tokens(pairs: list[tuple[str, float]], tokenizer: AutoTokenizer) -> list[tuple[str, float]]:
    special = set(tokenizer.all_special_tokens)
    cleaned = [
        (str(tok), float(score))
        for tok, score in pairs
        if str(tok).strip() and str(tok).strip() not in special
    ]
    return sorted(cleaned, key=lambda kv: abs(kv[1]), reverse=True)


# Above this many subword tokens, shap's max_evals="auto" budget grows large enough
# to make partition-explainer runtime impractical at dataset scale (tens of seconds
# per sample); below it, "auto" is already cheap and a fixed cap only adds overhead.
_AUTO_TOKEN_THRESHOLD = 10
_CAPPED_MAX_EVALS = 64


def explain_shap(text: str, max_evals: int | str = "auto") -> ExplanationResult:
    tokenizer, _ = _load_model()
    label, confidence = _predict(text)
    class_idx = LABELS.index(label)

    if max_evals == "auto" and len(tokenizer.tokenize(text)) > _AUTO_TOKEN_THRESHOLD:
        max_evals = _CAPPED_MAX_EVALS

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(_predict_proba, masker, output_names=LABELS)
    shap_values = explainer([text], max_evals=max_evals, silent=True)

    pairs = list(zip(shap_values.data[0], shap_values.values[0][:, class_idx]))
    tokens = _rank_tokens(pairs, tokenizer)
    return ExplanationResult(prediction=label, confidence=confidence, tokens=tokens)


# LIME's local surrogate model grows less reliable with fewer perturbation samples as
# the word count (feature count) rises; below the threshold a smaller budget is stable
# and much faster, at/above it the larger budget is kept to preserve reliability.
_LIME_WORD_THRESHOLD = 15
_LIME_SHORT_NUM_SAMPLES = 150
_LIME_LONG_NUM_SAMPLES = 300


def explain_lime(
    text: str,
    num_features: int = 15,
    num_samples: int | str = "auto",
    feature_selection: str = "auto",
    random_state: int | None = None,
) -> ExplanationResult:
    tokenizer, _ = _load_model()
    label, confidence = _predict(text)
    class_idx = LABELS.index(label)

    if num_samples == "auto":
        word_count = len(text.split())
        num_samples = (
            _LIME_SHORT_NUM_SAMPLES if word_count < _LIME_WORD_THRESHOLD else _LIME_LONG_NUM_SAMPLES
        )

    # LIME's default split_expression (\W+) treats Kannada vowel signs and virama as
    # non-word characters (Unicode categories Mc/Mn, not matched by \w), which shreds
    # Kannada words at every vowel sign. Splitting on whitespace instead keeps whole
    # words intact, since Kannada uses spaces between words like English.
    explainer = LimeTextExplainer(
        class_names=LABELS,
        split_expression=r"\s+",
        feature_selection=feature_selection,
        random_state=random_state,
    )
    exp = explainer.explain_instance(
        text,
        _predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=[class_idx],
    )
    pairs = exp.as_list(label=class_idx)
    tokens = _rank_tokens(pairs, tokenizer)
    return ExplanationResult(prediction=label, confidence=confidence, tokens=tokens)
