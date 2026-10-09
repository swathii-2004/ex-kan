import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.explainability import MODEL_VERSION, _load_model, explain_lime, explain_shap  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED / "explanations_output_v4.csv"
# Same files, same iteration order (high, medium, low) as generate_explanations.py's
# SAMPLE_FILES, so row N here is the same sentence as row N in explanations_output.csv
# (the v1 run) — required for row-for-row v1-vs-v4 comparison.
SAMPLE_FILES = {
    "high": PROCESSED / "explainability_sample_high.csv",
    "medium": PROCESSED / "explainability_sample_medium.csv",
    "low": PROCESSED / "explainability_sample_low.csv",
}
FIELDNAMES = [
    "text", "bucket", "label", "model_version", "prediction", "confidence",
    "shap_top_words", "shap_scores", "lime_top_words", "lime_scores",
]
PROGRESS_EVERY = 10


def load_all_rows(limit: int | None = None) -> list[dict]:
    rows = []
    for bucket, path in SAMPLE_FILES.items():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({"bucket": bucket, "label": r["label"], "text": r["text"]})
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_existing(output_path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """Read a possibly-interrupted output file: drop a truncated/corrupt final row
    (e.g. the process was killed mid-write), return the valid rows plus the
    (bucket, text) keys already completed. Raises if the file contains a different
    model_version than this run's — resuming across model versions would silently
    mix v1/v4 rows in one "v4" file."""
    if not output_path.exists():
        return [], set()

    with open(output_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], set()

        valid_raw_rows = []
        try:
            for raw_row in reader:
                if len(raw_row) != len(header):
                    # Truncated/corrupt row (partial write from a kill) — stop here;
                    # everything before this point is intact, this row and anything
                    # after it (if the file is this badly corrupt) is dropped.
                    break
                valid_raw_rows.append(raw_row)
        except csv.Error:
            pass  # stop at whatever parsed cleanly before the corruption

    valid_rows = [dict(zip(header, raw_row)) for raw_row in valid_raw_rows]

    bad_versions = {r.get("model_version") for r in valid_rows} - {MODEL_VERSION}
    if bad_versions:
        raise SystemExit(
            f"Refusing to resume {output_path}: contains model_version(s) {sorted(bad_versions)}, "
            f"expected only {MODEL_VERSION!r}. This file was not produced by the current model version."
        )

    done = {(r["bucket"], r["text"]) for r in valid_rows}
    return valid_rows, done


def read_battery() -> tuple[int | None, str | None]:
    matches = glob.glob("/sys/class/power_supply/BAT*")
    if not matches:
        return None, None
    bat_dir = Path(matches[0])
    try:
        capacity = int((bat_dir / "capacity").read_text().strip())
        status = (bat_dir / "status").read_text().strip()
        return capacity, status
    except (OSError, ValueError):
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing).")
    parser.add_argument(
        "--battery-threshold", type=int, default=30,
        help="Pause and exit if discharging at/below this battery percent (default 30).",
    )
    args = parser.parse_args()

    print(f"Active model_version: {MODEL_VERSION}", flush=True)
    if MODEL_VERSION != "v4":
        print(f"ABORT: MODEL_VERSION is {MODEL_VERSION!r}, not 'v4'. Set EX_KAN_MODEL_VERSION=v4 (or unset it, v4 is default).", flush=True)
        sys.exit(1)

    # Force the model to load now (and fail loudly now, per v4's fail-fast contract)
    # rather than partway through the batch.
    _load_model()
    print("Model loaded OK.", flush=True)

    all_rows = load_all_rows(limit=args.limit)
    existing_rows, done = load_existing(args.output)
    remaining = [r for r in all_rows if (r["bucket"], r["text"]) not in done]

    print(
        f"total rows: {len(all_rows)}  already done: {len(existing_rows)}  remaining: {len(remaining)}",
        flush=True,
    )

    completed_this_run = 0
    start = time.time()

    # Rewrite the file from the validated existing rows (this drops any truncated
    # tail from a prior kill), then append new rows from here.
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        f.flush()

        for row in remaining:
            text = row["text"]
            shap_result = explain_shap(text)
            lime_result = explain_lime(text)

            writer.writerow({
                "text": text,
                "bucket": row["bucket"],
                "label": row["label"],
                "model_version": MODEL_VERSION,
                "prediction": shap_result.prediction,
                "confidence": shap_result.confidence,
                "shap_top_words": json.dumps([t for t, _ in shap_result.tokens]),
                "shap_scores": json.dumps([s for _, s in shap_result.tokens]),
                "lime_top_words": json.dumps([t for t, _ in lime_result.tokens]),
                "lime_scores": json.dumps([s for _, s in lime_result.tokens]),
            })
            f.flush()

            completed_this_run += 1
            total_done = len(existing_rows) + completed_this_run

            if completed_this_run % PROGRESS_EVERY == 0:
                capacity, status = read_battery()
                elapsed = time.time() - start
                rate = completed_this_run / elapsed
                remaining_count = len(remaining) - completed_this_run
                eta_seconds = remaining_count / rate if rate > 0 else float("nan")
                batt_str = f"{capacity}%/{status}" if capacity is not None else "no battery"
                print(
                    f"PROGRESS: {total_done}/{len(all_rows)} done "
                    f"({completed_this_run} this run, {elapsed:.0f}s elapsed, "
                    f"{rate:.3f} rows/s, ETA {eta_seconds/60:.1f} min, battery {batt_str})",
                    flush=True,
                )

                if status == "Discharging" and capacity is not None and capacity <= args.battery_threshold:
                    f.flush()
                    print(
                        f"PAUSED for battery at {total_done}/{len(all_rows)} rows; "
                        f"rerun the same command to resume",
                        flush=True,
                    )
                    sys.exit(0)

    print(f"DONE: {len(existing_rows) + completed_this_run}/{len(all_rows)} total rows in {args.output}", flush=True)


if __name__ == "__main__":
    main()
