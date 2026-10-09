import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import MODEL_VERSION, _load_model, predict  # noqa: E402
from app.services.faithfulness import deletion_test, faithfulness_score, insertion_test  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_INPUT_PATH = PROCESSED / "explanations_output.csv"
DEFAULT_OUTPUT_PATH = PROCESSED / "faithfulness_results.csv"
FIELDNAMES = ["text", "bucket", "label", "model_version", "prediction", "lime_faithfulness_score", "shap_faithfulness_score"]
TOP_N = 5
PROGRESS_EVERY = 100


def load_done(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        rows = list(csv.DictReader(f))
    bad_versions = {r.get("model_version") for r in rows} - {MODEL_VERSION}
    if bad_versions:
        raise SystemExit(
            f"Refusing to resume {output_path}: contains model_version(s) {sorted(bad_versions)}, "
            f"expected only {MODEL_VERSION!r}."
        )
    return {(r["bucket"], r["text"]) for r in rows}


def load_input_rows(input_path: Path) -> list[dict]:
    with open(input_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    input_path, output_path = args.input, args.output

    print(f"Active model_version: {MODEL_VERSION}", flush=True)
    if MODEL_VERSION != "v4":
        print(f"ABORT: MODEL_VERSION is {MODEL_VERSION!r}, not 'v4'.", flush=True)
        sys.exit(1)

    tokenizer, _ = _load_model()

    done = load_done(output_path)
    all_rows = load_input_rows(input_path)
    remaining = [r for r in all_rows if (r["bucket"], r["text"]) not in done]

    print(f"total rows: {len(all_rows)}  already done: {len(done)}  remaining: {len(remaining)}", flush=True)

    is_new_file = not output_path.exists()
    completed_this_run = 0
    start = time.time()

    with open(output_path, "a", newline="") as f:
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
            lime_score = faithfulness_score(lime_deletion, lime_insertion)

            shap_deletion = deletion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
            shap_insertion = insertion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
            shap_score = faithfulness_score(shap_deletion, shap_insertion)

            writer.writerow({
                "text": text,
                "bucket": row["bucket"],
                "label": row["label"],
                "model_version": MODEL_VERSION,
                "prediction": row["prediction"],
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

    print(f"DONE: {len(done) + completed_this_run}/{len(all_rows)} total rows in {output_path}", flush=True)

    # --- final summary: average faithfulness score per bucket ---
    with open(output_path) as f:
        results = list(csv.DictReader(f))

    by_bucket = defaultdict(lambda: {"lime": [], "shap": []})
    for r in results:
        by_bucket[r["bucket"]]["lime"].append(float(r["lime_faithfulness_score"]))
        by_bucket[r["bucket"]]["shap"].append(float(r["shap_faithfulness_score"]))

    print("\n=== KEY RESULT: average faithfulness score per bucket ===", flush=True)
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
