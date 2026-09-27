import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.explainability import MODEL_PATH  # noqa: E402
from app.services.faithfulness import deletion_test, faithfulness_score, insertion_test  # noqa: E402

# Deterministic fake model so the test logic can be checked without the slow real
# model: "distress" if any negative keyword is present, confidence scales with hit count.
NEGATIVE_WORDS = {"kettadaguthide", "terrible"}

# The SHAP cases below use the real MuRIL tokenizer (loading only the tokenizer, not
# the model weights, is fast) so the subword segmentation being tested is the exact
# thing explain_shap's top_words would come from — not a hand-rolled fake tokenizer.
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    return _TOKENIZER


def fake_model(text: str) -> tuple[str, float]:
    if not text.strip():
        return "not-distress", 0.5
    hits = set(text.lower().split()) & NEGATIVE_WORDS
    if hits:
        return "distress", min(0.6 + 0.15 * len(hits), 0.95)
    return "not-distress", 0.85


CASES = [
    {
        "name": "faithful: top word is the actual driver",
        "text": "movie tumba kettadaguthide guru",
        "top_words": ["kettadaguthide"],
        "expect_deletion_faithful": True,
        "expect_insertion_faithful": True,
    },
    {
        "name": "unfaithful: top words are irrelevant filler",
        "text": "movie tumba kettadaguthide guru",
        "top_words": ["movie", "guru"],
        "expect_deletion_faithful": False,
        "expect_insertion_faithful": False,
    },
    {
        "name": "partial: top words capture only one of two drivers",
        "text": "movie tumba kettadaguthide mattu terrible guru",
        "top_words": ["terrible"],
        "expect_deletion_faithful": False,
        "expect_insertion_faithful": True,
    },
    {
        "name": "edge case: top words are the entire text",
        "text": "kettadaguthide",
        "top_words": ["kettadaguthide"],
        "expect_deletion_faithful": True,
        "expect_insertion_faithful": True,
    },
    {
        # MuRIL's tokenizer splits "kettadaguthide" into subword pieces
        # ['ket', 'tad', 'agu', 'thi', 'de ']. Reconstructing all of them should
        # behave exactly like the whole-word LIME case above.
        "name": "shap: full subword run reconstructs the whole driving word",
        "text": "movie tumba kettadaguthide guru",
        "top_words": ["ket", "tad", "agu", "thi", "de "],
        "method": "shap",
        "expect_deletion_faithful": True,
        "expect_insertion_faithful": True,
    },
    {
        # Only 2 of the 5 subword pieces of "kettadaguthide" — enough to break the
        # word on deletion (necessity holds), but not enough to reconstruct anything
        # recognizable on insertion (not sufficient on their own).
        "name": "shap: partial subword fragments break the word but aren't sufficient alone",
        "text": "movie tumba kettadaguthide guru",
        "top_words": ["tad", "thi"],
        "method": "shap",
        "expect_deletion_faithful": True,
        "expect_insertion_faithful": False,
    },
]


def main() -> None:
    for case in CASES:
        method = case.get("method", "lime")
        tokenizer = _tokenizer() if method == "shap" else None

        print(f"=== {case['name']} ===")
        print(f"text: {case['text']!r}  top_words: {case['top_words']}  method: {method}")

        deletion = deletion_test(case["text"], case["top_words"], fake_model, method=method, tokenizer=tokenizer)
        insertion = insertion_test(case["text"], case["top_words"], fake_model, method=method, tokenizer=tokenizer)
        score = faithfulness_score(deletion, insertion)

        print(
            f"  deletion:  {deletion.original_prediction}({deletion.original_confidence:.2f}) -> "
            f"{deletion.modified_prediction}({deletion.modified_confidence:.2f})  "
            f"modified_text={deletion.modified_text!r}  "
            f"flipped={deletion.prediction_flipped}  drop={deletion.confidence_drop:.2f}  "
            f"faithful={deletion.faithful}"
        )
        print(
            f"  insertion: {insertion.original_prediction}({insertion.original_confidence:.2f}) -> "
            f"{insertion.modified_prediction}({insertion.modified_confidence:.2f})  "
            f"modified_text={insertion.modified_text!r}  "
            f"held={insertion.prediction_held}  retained={insertion.confidence_retained:.2f}  "
            f"faithful={insertion.faithful}"
        )
        print(f"  faithfulness_score: {score:.3f}")

        assert deletion.faithful == case["expect_deletion_faithful"], (
            f"deletion.faithful mismatch for {case['name']!r}: "
            f"got {deletion.faithful}, expected {case['expect_deletion_faithful']}"
        )
        assert insertion.faithful == case["expect_insertion_faithful"], (
            f"insertion.faithful mismatch for {case['name']!r}: "
            f"got {insertion.faithful}, expected {case['expect_insertion_faithful']}"
        )
        print("  PASS")
        print()

    print("All test cases passed.")


if __name__ == "__main__":
    main()
