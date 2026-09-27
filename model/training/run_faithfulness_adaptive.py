import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import _load_model, predict  # noqa: E402
from app.services.faithfulness import deletion_test, faithfulness_score, insertion_test  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "explanations_output.csv"
OUTPUT_PATH = PROCESSED / "faithfulness_results_adaptive.csv"
FIELDNAMES = ["text", "bucket", "label", "prediction", "top_n", "lime_faithfulness_score", "shap_faithfulness_score"]
PROGRESS_EVERY = 100

# Length-proportional top-N: fixes the fixed-top-5 confound, where a flat N touched a
# much larger fraction of short sentences than long ones, mechanically biasing
# deletion/insertion results against longer text independent of explanation quality.
TOP_N_FRACTION = 0.25
TOP_N_MIN = 3
TOP_N_MAX = 10


def adaptive_top_n(word_count: int) -> int:
    return max(TOP_N_MIN, min(TOP_N_MAX, round(word_count * TOP_N_FRACTION)))


def load_done() -> set[tuple[str, str]]:
    if not OUTPUT_PATH.exists():
        return set()
    with open(OUTPUT_PATH) as f:
        return {(r["bucket"], r["text"]) for r in csv.DictReader(f)}


def load_input_rows() -> list[dict]:
    with open(INPUT_PATH) as f:
        return list(csv.DictReader(f))


def main() -> None:
    tokenizer, _ = _load_model()

    done = load_done()
    all_rows = load_input_rows()
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
            top_n = adaptive_top_n(len(text.split()))
            lime_words = json.loads(row["lime_top_words"])[:top_n]
            shap_words = json.loads(row["shap_top_words"])[:top_n]

            lime_deletion = deletion_test(text, lime_words, predict, method="lime")
            lime_insertion = insertion_test(text, lime_words, predict, method="lime")
            lime_score = faithfulness_score(lime_deletion, lime_insertion)

            shap_deletion = deletion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
            shap_insertion = insertion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
            shap_score = faithfulness_score(shap_deletion, shap_insertion)

            writer.writerow({
                "text": text,
                "bucket": row["bucket"],
                "label": row["label"],
                "prediction": row["prediction"],
                "top_n": top_n,
                "lime_faithfulness_score": lime_score,
                "shap_faithfulness_score": shap_score,
            })
            f.flush()

            completed_this_run += 1
            total_done = len(done) + completed_this_run
            if completed_this_run % PROGRESS_EVERY == 0 or completed_this_run in (20, 30):
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

    # --- final summary: average faithfulness score per bucket ---
    with open(OUTPUT_PATH) as f:
        results = list(csv.DictReader(f))

    by_bucket = defaultdict(lambda: {"lime": [], "shap": []})
    for r in results:
        by_bucket[r["bucket"]]["lime"].append(float(r["lime_faithfulness_score"]))
        by_bucket[r["bucket"]]["shap"].append(float(r["shap_faithfulness_score"]))

    print("\n=== KEY RESULT (length-proportional top-N): average faithfulness score per bucket ===", flush=True)
    for bucket in ("low", "medium", "high"):
        if bucket not in by_bucket:
            continue
        lime_scores = by_bucket[bucket]["lime"]
        shap_scores = by_bucket[bucket]["shap"]
        print(
            f"{bucket:8s} n={len(lime_scores):3d}  "
            f"LIME avg={sum(lime_scores)/len(lime_scores):.4f}  "
            f"SHAP avg={sum(shap_scores)/len(shap_scores):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
