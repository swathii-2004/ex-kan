"""
Improved code-mix ratio calculator + bucketer for DravidianCodeMix Kannada sentiment data.

This REPLACES the old script-only bucketer (code_mix_bucketer.py), which was blind to
Romanized Kannada (Kannada written in Latin letters) and wrongly dumped ~91% of rows
into "Low". This version detects Romanized Kannada via an English-dictionary check,
with noise filters to avoid false positives (proper nouns, spam, digits, slang).

Validated methodology (matches earlier analysis exactly):
- effective_score = 2 * min(kannada_language_pct, english_pct)
  where kannada_language_pct = kannada_script_pct + romanized_kannada_pct
- Thresholds: Low < 0.20, Medium 0.20-0.60, High >= 0.60
- Expected output on the ~7,655-row base dataset: Low ~3,472, Medium ~1,565, High ~2,614

Run: python improved_bucketer.py
Requires: the DravidianCodeMix-2020.zip (same as used by the original code_mix_bucketer.py)
          and /usr/share/dict/words (standard on most Linux systems; install
          `words` package via `sudo apt install wamerican` if missing).
"""
import csv
import os
import re
import zipfile
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this path to wherever DravidianCodeMix-Dataset-main/ actually sits relative
# to this script. Update if your folder layout differs.
ZIP_PATH = os.path.join(BASE_DIR, "DravidianCodeMix-Dataset-main", "DravidianCodeMix-2020.zip")
INNER_PATH = "DravidianCodeMix/kannada_sentiment.csv"
DICT_PATH = "/usr/share/dict/words"
OUT_DIR = BASE_DIR

KANNADA_RE = re.compile(r'[\u0C80-\u0CFF]')
LATIN_RE = re.compile(r'[A-Za-z]')
DIGIT_RE = re.compile(r'\d')
REPEATED_CHAR_RE = re.compile(r'(.)\1{2,}')  # 3+ repeated characters (e.g. "soooo")
WORD_CLEAN_RE = re.compile(r"[^a-z']")

# Internet slang whitelist — words missing from formal dictionaries that are genuinely
# English, not Romanized Kannada. Extend this list if you spot more false positives.
SLANG_WHITELIST = {
    "bro", "bros", "pls", "plz", "u", "ur", "thx", "lol", "omg", "gonna", "wanna",
    "gotta", "kinda", "sorta", "yeah", "yep", "nah", "ok", "okay", "hey", "hii", "hi",
    "bday", "insta", "fb", "whatsapp", "youtube", "yt", "dm", "pm", "asap", "btw",
    "fyi", "tbh", "imo", "lmao", "lmfao", "rofl", "wtf", "idk", "smh", "fr", "ngl",
}

# Thresholds on the combined effective_score (0 = monolingual, 1 = perfectly balanced)
LOW_MAX = 0.20
MEDIUM_MAX = 0.60
MIN_SIGNAL_TOKENS = 4  # rows with fewer signal tokens are flagged low_confidence


def load_english_words(path):
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words | SLANG_WHITELIST


def is_noise_token(tok, position_in_sentence):
    """Tokens excluded from the romanized-Kannada-candidate count entirely."""
    if DIGIT_RE.search(tok):
        return True
    if REPEATED_CHAR_RE.search(tok):
        return True
    # Title-Case or ALL-CAPS tokens NOT at sentence start are likely proper nouns
    # (names, places, celebrities) rather than Romanized Kannada.
    if position_in_sentence > 0 and len(tok) > 1:
        if tok.isupper() or (tok[0].isupper() and tok[1:].islower()):
            return True
    return False


def token_script(tok):
    if KANNADA_RE.search(tok):
        return "kannada"
    if LATIN_RE.search(tok):
        return "latin"
    return "other"


def analyze(label, text, english_words):
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return None

    kannada_tokens = 0
    english_tokens = 0
    romanized_candidate_tokens = 0
    noise_tokens = 0

    for i, tok in enumerate(tokens):
        script = token_script(tok)
        if script == "kannada":
            kannada_tokens += 1
        elif script == "latin":
            if is_noise_token(tok, i):
                noise_tokens += 1
                continue
            cleaned = WORD_CLEAN_RE.sub("", tok.lower())
            if cleaned and cleaned in english_words:
                english_tokens += 1
            else:
                romanized_candidate_tokens += 1
        # "other" (pure punctuation/emoji/numerals-only) tokens are not counted at all

    denom = kannada_tokens + english_tokens + romanized_candidate_tokens
    if denom == 0:
        return {
            "label": label, "text": text, "n_tokens": n,
            "kannada_language_pct": 0.0, "english_pct": 0.0,
            "effective_score": 0.0, "low_confidence": True,
            "bucket": "unclassifiable",
        }

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

    return {
        "label": label,
        "text": text,
        "n_tokens": n,
        "kannada_script_pct": round(kannada_script_pct, 4),
        "romanized_kannada_pct": round(romanized_kannada_pct, 4),
        "kannada_language_pct": round(kannada_language_pct, 4),
        "english_pct": round(english_pct, 4),
        "effective_score": round(effective_score, 4),
        "low_confidence": low_confidence,
        "bucket": bucket,
    }


def load_base_rows():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(INNER_PATH) as f:
            text = f.read().decode("utf-8")
    reader = csv.reader(text.splitlines(), delimiter="\t")
    return [(r[0], r[1]) for r in reader if len(r) == 2]


def main():
    english_words = load_english_words(DICT_PATH)
    raw_rows = load_base_rows()
    print(f"Loaded {len(raw_rows)} rows from {INNER_PATH}")

    results = [analyze(label, text, english_words) for label, text in raw_rows]
    results = [r for r in results if r is not None]

    buckets = {"low": [], "medium": [], "high": [], "unclassifiable": []}
    for r in results:
        buckets[r["bucket"]].append(r)

    fieldnames = [
        "label", "text", "n_tokens", "kannada_script_pct", "romanized_kannada_pct",
        "kannada_language_pct", "english_pct", "effective_score", "low_confidence", "bucket",
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    for b in ("low", "medium", "high"):
        path = os.path.join(OUT_DIR, f"{b}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in buckets[b]:
                row = {k: r.get(k, "") for k in fieldnames}
                w.writerow(row)

    total = len(results)
    total_bucketed = total - len(buckets["unclassifiable"])

    print(f"\nTotal rows scored: {total} | Unclassifiable: {len(buckets['unclassifiable'])}")
    print("=" * 90)
    print(f"{'Bucket':<10}{'Count':>8}{'% of scored':>14}")
    print("=" * 90)
    for b in ("low", "medium", "high"):
        cnt = len(buckets[b])
        pct = 100 * cnt / total_bucketed if total_bucketed else 0
        label_counts = collections.Counter(r["label"] for r in buckets[b])
        label_str = ", ".join(f"{k}={v}" for k, v in label_counts.most_common())
        print(f"{b.capitalize():<10}{cnt:>8}{pct:>13.1f}%   {label_str}")
    print(f"\nCSV files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
