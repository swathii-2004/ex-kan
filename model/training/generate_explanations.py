import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import explain_lime, explain_shap  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED / "explanations_output.csv"
SAMPLE_FILES = {
    "high": PROCESSED / "explainability_sample_high.csv",
    "medium": PROCESSED / "explainability_sample_medium.csv",
    "low": PROCESSED / "explainability_sample_low.csv",
}
FIELDNAMES = [
    "text", "bucket", "label", "prediction", "confidence",
    "shap_top_words", "shap_scores", "lime_top_words", "lime_scores",
]
PROGRESS_EVERY = 50


def load_done() -> set[tuple[str, str]]:
    if not OUTPUT_PATH.exists():
        return set()
    with open(OUTPUT_PATH) as f:
        return {(r["bucket"], r["text"]) for r in csv.DictReader(f)}


def load_all_rows() -> list[dict]:
    rows = []
    for bucket, path in SAMPLE_FILES.items():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({"bucket": bucket, "label": r["label"], "text": r["text"]})
    return rows


def main() -> None:
    done = load_done()
    all_rows = load_all_rows()
    remaining = [r for r in all_rows if (r["bucket"], r["text"]) not in done]

    print(f"total rows: {len(all_rows)}  already done: {len(done)}  remaining: {len(remaining)}", flush=True)

    is_new_file = not OUTPUT_PATH.exists()
    completed_this_run = 0
    start = time.time()

    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
            f.flush()

        for row in remaining:
            text = row["text"]
            shap_result = explain_shap(text)
            lime_result = explain_lime(text)

            writer.writerow({
                "text": text,
                "bucket": row["bucket"],
                "label": row["label"],
                "prediction": shap_result.prediction,
                "confidence": shap_result.confidence,
                "shap_top_words": json.dumps([t for t, _ in shap_result.tokens]),
                "shap_scores": json.dumps([s for _, s in shap_result.tokens]),
                "lime_top_words": json.dumps([t for t, _ in lime_result.tokens]),
                "lime_scores": json.dumps([s for _, s in lime_result.tokens]),
            })
            f.flush()

            completed_this_run += 1
            total_done = len(done) + completed_this_run
            if completed_this_run % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                rate = completed_this_run / elapsed
                remaining_count = len(remaining) - completed_this_run
                eta_seconds = remaining_count / rate if rate > 0 else float("nan")
                print(
                    f"PROGRESS: {total_done}/{len(all_rows)} done "
                    f"({completed_this_run} this run, {elapsed:.0f}s elapsed, "
                    f"{rate:.3f} rows/s, ETA {eta_seconds/60:.1f} min)",
                    flush=True,
                )

    print(f"DONE: {len(done) + completed_this_run}/{len(all_rows)} total rows in {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
