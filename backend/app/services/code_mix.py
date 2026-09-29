from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Reconstructed from data/processed/{low,medium,high}.csv, since the original
# improved_bucketer.py script referenced in PROJECT_ROADMAP.md isn't checked into the
# repo. The effective_score formula and bucket thresholds below were reverse-engineered
# and exactly match every row across all three bucket files:
#   effective_score = 2 * min(kannada_language_pct, english_pct) / (kannada_language_pct + english_pct)
#   low:    effective_score in [0.0,   0.2)   (observed range 0.0-0.1923)
#   medium: effective_score in [0.2,   0.6)   (observed range 0.2-0.5926)
#   high:   effective_score in [0.6,   1.0]   (observed range 0.6-1.0)
# The per-token classification (kannada script / romanized-Kannada / English) is a
# best-effort reconstruction, not verified against the original script, since a token
# is presumed "romanized Kannada" whenever it isn't recognizable English or Kannada
# script — a reasonable default for this Kannada-English corpus, but an approximation.

LOW_HIGH_THRESHOLD = 0.2
MEDIUM_HIGH_THRESHOLD = 0.6

_KANNADA_RE = re.compile(r"[ಀ-೿]")
_TOKEN_RE = re.compile(r"[A-Za-zಀ-೿']+")


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    with open("/usr/share/dict/words") as f:
        return frozenset(w.strip().lower() for w in f if w.strip().isalpha())


def _is_english_word(token: str) -> bool:
    t = token.strip("'").lower()
    return len(t) >= 2 and t.isalpha() and t in _english_words()


@dataclass
class CodeMixResult:
    bucket: str
    effective_score: float
    kannada_script_pct: float
    romanized_kannada_pct: float
    english_pct: float


def analyze_code_mix(text: str) -> CodeMixResult:
    tokens = _TOKEN_RE.findall(text)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return CodeMixResult(bucket="low", effective_score=0.0, kannada_script_pct=0.0,
                              romanized_kannada_pct=0.0, english_pct=0.0)

    kannada_script_count = sum(1 for t in tokens if _KANNADA_RE.search(t))
    english_count = sum(1 for t in tokens if not _KANNADA_RE.search(t) and _is_english_word(t))
    romanized_count = n_tokens - kannada_script_count - english_count

    kannada_script_pct = kannada_script_count / n_tokens
    romanized_kannada_pct = romanized_count / n_tokens
    english_pct = english_count / n_tokens
    kannada_language_pct = kannada_script_pct + romanized_kannada_pct

    denom = kannada_language_pct + english_pct
    effective_score = 2 * min(kannada_language_pct, english_pct) / denom if denom > 0 else 0.0

    if effective_score < LOW_HIGH_THRESHOLD:
        bucket = "low"
    elif effective_score < MEDIUM_HIGH_THRESHOLD:
        bucket = "medium"
    else:
        bucket = "high"

    return CodeMixResult(
        bucket=bucket,
        effective_score=round(effective_score, 4),
        kannada_script_pct=round(kannada_script_pct, 4),
        romanized_kannada_pct=round(romanized_kannada_pct, 4),
        english_pct=round(english_pct, 4),
    )
