import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

VALID_LABELS = ("Positive", "Negative", "Mixed feelings")
SEED = 20260927
SAMPLE_SIZE = 300

SOURCES = {
    "high": PROCESSED / "CLEAN_high.csv",
    "medium": PROCESSED / "CLEAN_medium.csv",
    "low": PROCESSED / "low.csv",
}


def load_pool(path: Path) -> list[dict]:
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["label"] in VALID_LABELS]


def stratified_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    by_label = defaultdict(list)
    for r in pool:
        by_label[r["label"]].append(r)

    total = len(pool)
    raw_alloc = {lbl: n * len(rows) / total for lbl, rows in by_label.items()}
    floor_alloc = {lbl: int(v) for lbl, v in raw_alloc.items()}
    remainder = n - sum(floor_alloc.values())
    fracs = sorted(raw_alloc.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
    for lbl, _ in fracs[:remainder]:
        floor_alloc[lbl] += 1

    rng = random.Random(seed)
    sample = []
    for lbl, count in floor_alloc.items():
        sample.extend(rng.sample(by_label[lbl], count))
    rng.shuffle(sample)
    return sample


def main() -> None:
    fieldnames = [
        "label", "text", "n_tokens", "kannada_script_pct", "romanized_kannada_pct",
        "kannada_language_pct", "english_pct", "effective_score", "low_confidence", "bucket",
    ]
    for bucket, path in SOURCES.items():
        pool = load_pool(path)
        n = min(SAMPLE_SIZE, len(pool))
        sample = stratified_sample(pool, n, SEED)
        out_path = PROCESSED / f"explainability_sample_{bucket}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in sample:
                w.writerow({k: r[k] for k in fieldnames})
        print(f"{bucket}: pool={len(pool)} sampled={len(sample)} -> {out_path}")
        print(f"  label dist: {Counter(r['label'] for r in sample)}")


if __name__ == "__main__":
    main()
