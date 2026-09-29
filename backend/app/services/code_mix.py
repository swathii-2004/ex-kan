from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Ported verbatim (tokenization, noise filtering, slang whitelist, scoring, thresholds)
# from the original improved_bucketer.py, recovered from ~/Downloads after not having
# been checked into this repo. Validated to reproduce data/processed/{low,medium,high}.csv
# exactly (0 mismatches across 1,500 real rows checked). Only change from the original:
# adapted from a batch CSV-in/CSV-out script to a single-text analysis function.

LOW_MAX = 0.20
MEDIUM_MAX = 0.60
MIN_SIGNAL_TOKENS = 4

_KANNADA_RE = re.compile(r"[ಀ-೿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ repeated characters (e.g. "soooo")
_WORD_CLEAN_RE = re.compile(r"[^a-z']")

# Internet slang whitelist — words missing from formal dictionaries that are genuinely
# English, not Romanized Kannada.
_SLANG_WHITELIST = {
    "bro", "bros", "pls", "plz", "u", "ur", "thx", "lol", "omg", "gonna", "wanna",
    "gotta", "kinda", "sorta", "yeah", "yep", "nah", "ok", "okay", "hey", "hii", "hi",
    "bday", "insta", "fb", "whatsapp", "youtube", "yt", "dm", "pm", "asap", "btw",
    "fyi", "tbh", "imo", "lmao", "lmfao", "rofl", "wtf", "idk", "smh", "fr", "ngl",
}


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as f:
        words = {line.strip().lower() for line in f if line.strip()}
    return frozenset(words | _SLANG_WHITELIST)


def _is_noise_token(tok: str, position_in_sentence: int) -> bool:
    """Tokens excluded from the romanized-Kannada-candidate count entirely."""
    if _DIGIT_RE.search(tok):
        return True
    if _REPEATED_CHAR_RE.search(tok):
        return True
    # Title-Case or ALL-CAPS tokens not at sentence start are likely proper nouns
    # (names, places, celebrities) rather than Romanized Kannada.
    if position_in_sentence > 0 and len(tok) > 1:
        if tok.isupper() or (tok[0].isupper() and tok[1:].islower()):
            return True
    return False


def _token_script(tok: str) -> str:
    if _KANNADA_RE.search(tok):
        return "kannada"
    if _LATIN_RE.search(tok):
        return "latin"
    return "other"


@dataclass
class CodeMixResult:
    bucket: str
    effective_score: float
    kannada_script_pct: float
    romanized_kannada_pct: float
    english_pct: float
    low_confidence: bool


def analyze_code_mix(text: str) -> CodeMixResult:
    english_words = _english_words()
    tokens = text.split()

    kannada_tokens = 0
    english_tokens = 0
    romanized_candidate_tokens = 0

    for i, tok in enumerate(tokens):
        script = _token_script(tok)
        if script == "kannada":
            kannada_tokens += 1
        elif script == "latin":
            if _is_noise_token(tok, i):
                continue
            cleaned = _WORD_CLEAN_RE.sub("", tok.lower())
            if cleaned and cleaned in english_words:
                english_tokens += 1
            else:
                romanized_candidate_tokens += 1
        # "other" (pure punctuation/emoji/numerals-only) tokens are not counted at all

    denom = kannada_tokens + english_tokens + romanized_candidate_tokens
    if denom == 0:
        return CodeMixResult(bucket="low", effective_score=0.0, kannada_script_pct=0.0,
                              romanized_kannada_pct=0.0, english_pct=0.0, low_confidence=True)

    kannada_script_pct = kannada_tokens / denom
    romanized_kannada_pct = romanized_candidate_tokens / denom
    kannada_language_pct = kannada_script_pct + romanized_kannada_pct
    english_pct = english_tokens / denom

    effective_score = 2 * min(kannada_language_pct, english_pct)
    low_confidence = denom < MIN_SIGNAL_TOKENS

    if effective_score < LOW_MAX:
        bucket = "low"
    elif effective_score < MEDIUM_MAX:
        bucket = "medium"
    else:
        bucket = "high"

    return CodeMixResult(
        bucket=bucket,
        effective_score=round(effective_score, 4),
        kannada_script_pct=round(kannada_script_pct, 4),
        romanized_kannada_pct=round(romanized_kannada_pct, 4),
        english_pct=round(english_pct, 4),
        low_confidence=low_confidence,
    )
