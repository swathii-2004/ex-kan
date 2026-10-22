"""
EXPLORATORY analysis of the v1-vs-v4 faithfulness comparison — run once, no tuning.
Unlike analyze_faithfulness_stepc.py (primary/sensitivity) and
analyze_faithfulness_robustness.py (pre-specified confirmatory checks), this script
is exploratory: it looks at faithfulness broken down by word-count tertile, on top of
bucket, to see whether the bucket pattern holds within length strata or is confounded
by it. Results here are descriptive, not confirmatory.

1. Per-bucket word-count distribution (min, Q1, median, Q3, max) for the 900 rows.
2. Pooled word-count tertiles (3 bins across all 900 rows); for each (bin, bucket)
   cell: n and mean LIME/SHAP faithfulness (v1, v4) with cluster-bootstrap 95% CIs
   (resample unique texts, 10,000 resamples, seed 42). Cells with n < 30 are marked.
3. Within each bin: Medium-Low and Medium-High differences with CIs, for v1 and v4.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
N_RESAMPLES = 10_000
SEED = 42
BUCKETS = ["low", "medium", "high"]
METRICS = {"lime_faithfulness_score": "LIME", "shap_faithfulness_score": "SHAP"}
MIN_CELL_N = 30

rng = np.random.default_rng(SEED)


def build_dataset(adaptive_path: Path, expl_path: Path) -> pd.DataFrame:
    adaptive = pd.read_csv(adaptive_path)
    expl = pd.read_csv(expl_path)
    assert len(adaptive) == len(expl)
    assert (adaptive["text"].values == expl["text"].values).all()
    assert (adaptive["bucket"].values == expl["bucket"].values).all()
    df = adaptive.copy()
    df["word_count"] = df["text"].str.split().str.len()
    return df


v1 = build_dataset(PROCESSED / "faithfulness_results_adaptive.csv", PROCESSED / "explanations_output.csv")
v4 = build_dataset(PROCESSED / "faithfulness_results_adaptive_v4.csv", PROCESSED / "explanations_output_v4.csv")
assert (v1["text"].values == v4["text"].values).all() and (v1["bucket"].values == v4["bucket"].values).all()
assert (v1["word_count"].values == v4["word_count"].values).all()  # word_count depends only on text

# Pooled tertile bins (word_count is identical across v1/v4, so one set of bin edges
# applies to both).
bin_labels, bin_edges = pd.qcut(v1["word_count"], q=3, labels=["T1 (short)", "T2 (mid)", "T3 (long)"], retbins=True)
v1["wc_bin"] = bin_labels
v4["wc_bin"] = bin_labels.values
BINS = ["T1 (short)", "T2 (mid)", "T3 (long)"]

DATASETS = {"v1": v1, "v4": v4}


def cluster_boot_means(df: pd.DataFrame, value_col: str, rng: np.random.Generator, n_resamples: int = N_RESAMPLES) -> np.ndarray:
    groups = {t: g[value_col].to_numpy() for t, g in df.groupby("text")}
    texts = np.array(list(groups.keys()), dtype=object)
    n_texts = len(texts)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = texts[rng.integers(0, n_texts, size=n_texts)]
        vals = np.concatenate([groups[t] for t in sampled])
        boot_means[i] = vals.mean()
    return boot_means


def ci(arr: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(arr, [2.5, 97.5]))


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]"


print("EXPLORATORY ANALYSIS — faithfulness by word-count tertile x bucket (not a confirmatory test)")

print()
print("=" * 100)
print("EXPLORATORY ITEM 1: Per-bucket word-count distribution (900 rows)")
print("=" * 100)

print(f"\n{'bucket':8s} {'n':>5s} {'min':>6s} {'Q1':>6s} {'median':>8s} {'Q3':>6s} {'max':>6s}")
for bucket in BUCKETS:
    bdf = v1[v1["bucket"] == bucket]
    wc = bdf["word_count"]
    print(
        f"{bucket:8s} {len(bdf):5d} {wc.min():6.0f} {wc.quantile(0.25):6.1f} "
        f"{wc.median():8.1f} {wc.quantile(0.75):6.1f} {wc.max():6.0f}"
    )

print(f"\npooled tertile edges (word_count): {np.round(bin_edges, 2).tolist()}")


print()
print("=" * 100)
print("EXPLORATORY ITEM 2: Mean faithfulness per (word-count bin, bucket), v1 and v4, cluster-bootstrap CIs")
print("(cells with n < 30 are marked with *)")
print("=" * 100)

cell_boot = {}  # (version, bin, bucket, metric) -> boot array

for version, df in DATASETS.items():
    print(f"\n--- {version} ---")
    print(f"{'bin':12s} {'bucket':8s} {'metric':6s} {'n':>5s}  {'mean [95% CI]'}")
    for b in BINS:
        for bucket in BUCKETS:
            cell = df[(df["wc_bin"] == b) & (df["bucket"] == bucket)]
            n = len(cell)
            flag = " *" if n < MIN_CELL_N else ""
            for metric_col, label in METRICS.items():
                if n == 0:
                    print(f"{b:12s} {bucket:8s} {label:6s} {n:5d}  (no rows)")
                    continue
                boot = cluster_boot_means(cell, metric_col, rng)
                cell_boot[(version, b, bucket, metric_col)] = boot
                mean = cell[metric_col].mean()
                lo, hi = ci(boot)
                print(f"{b:12s} {bucket:8s} {label:6s} {n:5d}{flag}  {fmt_ci(mean, lo, hi)}")


print()
print("=" * 100)
print("EXPLORATORY ITEM 3: Within each word-count bin, Medium-Low and Medium-High (v1, v4)")
print("=" * 100)

for b in BINS:
    print(f"\n--- bin: {b} ---")
    for version in ("v1", "v4"):
        for metric_col, label in METRICS.items():
            med = cell_boot.get((version, b, "medium", metric_col))
            low = cell_boot.get((version, b, "low", metric_col))
            high = cell_boot.get((version, b, "high", metric_col))
            if med is None or low is None or high is None:
                print(f"  {version} {label:6s}: insufficient data in this bin for one or more buckets")
                continue
            d_ml = med - low
            d_mh = med - high
            print(f"  {version} {label:6s} Medium-Low : {fmt_ci(d_ml.mean(), *ci(d_ml))}")
            print(f"  {version} {label:6s} Medium-High: {fmt_ci(d_mh.mean(), *ci(d_mh))}")

print("\nDONE")
