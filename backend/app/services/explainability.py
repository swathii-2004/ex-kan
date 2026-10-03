from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

_CHECKPOINTS_DIR = Path(__file__).resolve().parents[3] / "model" / "checkpoints"
MODEL_V1_PATH = _CHECKPOINTS_DIR / "ex_kan_model"
MODEL_V2_PATH = _CHECKPOINTS_DIR / "ex_kan_model_v2"
# Backwards-compat alias (test_faithfulness.py only needs a MuRIL tokenizer, which is
# identical between v1/v2, so it keeps pointing at v1).
MODEL_PATH = MODEL_V1_PATH

MURIL_BASE = "google/muril-base-cased"
_LSTM_HIDDEN_DIM = 128
_DROPOUT = 0.3

# Which checkpoint /predict and /explain serve. v2 (MuRIL + BiLSTM) is the improved
# model trained after v1 (a plain MuRIL/BERT fine-tune); set EX_KAN_MODEL_VERSION=v1
# in the environment to force the older checkpoint.
MODEL_VERSION = os.environ.get("EX_KAN_MODEL_VERSION", "v2")

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


class MurilLSTMClassifier(nn.Module):
    """v2 architecture: MuRIL encoder -> bidirectional LSTM over the token sequence
    -> dropout -> linear head on the concatenated final forward/backward hidden
    states. Layer names/shapes match model_state.pt's state_dict exactly (bert.*,
    lstm.{weight,bias}_{ih,hh}_l0[_reverse], classifier.{weight,bias})."""

    def __init__(self, muril_base: str = MURIL_BASE) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(muril_base)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=_LSTM_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(_DROPOUT)
        self.classifier = nn.Linear(_LSTM_HIDDEN_DIM * 2, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object) -> SequenceClassifierOutput:
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        _, (hidden, _) = self.lstm(sequence_output)
        # hidden: (num_directions, batch, hidden_dim) for a single-layer LSTM — index 0
        # is the final forward step, index 1 the final backward step.
        pooled = torch.cat((hidden[0], hidden[1]), dim=-1)
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)


def _load_v1() -> tuple[AutoTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_V1_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_V1_PATH)
    model.eval()
    return tokenizer, model


def _load_v2() -> tuple[AutoTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_V2_PATH)
    model = MurilLSTMClassifier()
    state_dict = torch.load(MODEL_V2_PATH / "model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoTokenizer, nn.Module]:
    if MODEL_VERSION == "v1":
        return _load_v1()
    try:
        return _load_v2()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any v2 load failure should fall back, not crash the API
        warnings.warn(f"Failed to load v2 model ({exc!r}); falling back to v1.", stacklevel=2)
        return _load_v1()


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


def predict(text: str) -> tuple[str, float]:
    """Public PredictFn-shaped wrapper around the real trained model, for use as the
    `model` callable passed to faithfulness.deletion_test/insertion_test."""
    return _predict(text)


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
