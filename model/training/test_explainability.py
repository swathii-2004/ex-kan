import csv
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import explain_lime, explain_shap  # noqa: E402

BUCKETS = {
    "low": REPO_ROOT / "data" / "processed" / "low.csv",
    "medium": REPO_ROOT / "data" / "processed" / "CLEAN_medium.csv",
    "high": REPO_ROOT / "data" / "processed" / "CLEAN_high.csv",
}
VALID_LABELS = {"Positive", "Negative", "Mixed feelings"}
SAMPLES_PER_BUCKET = 5
SEED = 42


def load_samples(path: Path, n: int) -> list[str]:
    with open(path) as f:
        rows = [r["text"] for r in csv.DictReader(f) if r["label"] in VALID_LABELS]
    random.Random(SEED).shuffle(rows)
    return rows[:n]


def format_tokens(tokens: list[tuple[str, float]], top_n: int = 5) -> str:
    return ", ".join(f"{tok!r} ({score:+.3f})" for tok, score in tokens[:top_n])


def main() -> None:
    for bucket, path in BUCKETS.items():
        print(f"\n{'=' * 80}\nBUCKET: {bucket.upper()}\n{'=' * 80}")
        for text in load_samples(path, SAMPLES_PER_BUCKET):
            print(f"\nTEXT: {text}")

            t0 = time.time()
            shap_result = explain_shap(text)
            shap_time = time.time() - t0

            t0 = time.time()
            lime_result = explain_lime(text)
            lime_time = time.time() - t0

            print(
                f"  Prediction: {shap_result.prediction} "
                f"(SHAP conf={shap_result.confidence:.3f}, LIME conf={lime_result.confidence:.3f})"
            )
            print(f"  SHAP top-5 ({shap_time:.2f}s): {format_tokens(shap_result.tokens)}")
            print(f"  LIME top-5 ({lime_time:.2f}s): {format_tokens(lime_result.tokens)}")


if __name__ == "__main__":
    main()
