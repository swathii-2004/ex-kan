import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import _load_model, predict  # noqa: E402
from app.services.faithfulness import deletion_test, insertion_test  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "explanations_output.csv"
OUTPUT_PATH = PROCESSED / "faithfulness_components.csv"
TOP_N = 5
PROGRESS_EVERY = 100

FIELDNAMES = [
    "text", "bucket", "label", "prediction",
    "lime_confidence_drop", "lime_prediction_flipped",
    "lime_confidence_retained", "lime_prediction_held",
    "shap_confidence_drop", "shap_prediction_flipped",
    "shap_confidence_retained", "shap_prediction_held",
]


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
            lime_words = json.loads(row["lime_top_words"])[:TOP_N]
            shap_words = json.loads(row["shap_top_words"])[:TOP_N]

            lime_deletion = deletion_test(text, lime_words, predict, method="lime")
            lime_insertion = insertion_test(text, lime_words, predict, method="lime")

            shap_deletion = deletion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
            shap_insertion = insertion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)

            writer.writerow({
                "text": text,
                "bucket": row["bucket"],
                "label": row["label"],
                "prediction": row["prediction"],
                "lime_confidence_drop": lime_deletion.confidence_drop,
                "lime_prediction_flipped": lime_deletion.prediction_flipped,
                "lime_confidence_retained": lime_insertion.confidence_retained,
                "lime_prediction_held": lime_insertion.prediction_held,
                "shap_confidence_drop": shap_deletion.confidence_drop,
                "shap_prediction_flipped": shap_deletion.prediction_flipped,
                "shap_confidence_retained": shap_insertion.confidence_retained,
                "shap_prediction_held": shap_insertion.prediction_held,
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


if __name__ == "__main__":
    main()
