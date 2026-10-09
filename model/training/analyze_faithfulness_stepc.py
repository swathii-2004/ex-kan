"""
Step C analysis: v1-vs-v4 faithfulness comparison on the paired 900-row sample.

1. Primary analysis (all 900 rows): per-bucket mean faithfulness with cluster
   bootstrap 95% CIs (resample unique texts within bucket), plus Medium-Low and
   Medium-High contrasts, for v1 and v4.
2. Sensitivity analysis: same, but on unique (text, bucket) rows only (duplicates
   averaged).
3. Within-text variability for texts appearing >=3 times, and the Low-bucket mean
   with/without the 13x-repeated comment.
4. Paired v1-vs-v4 mean difference per bucket (cluster bootstrap on the per-row
   v4-v1 difference).
5. Pearson/Spearman correlation between effective_score and faithfulness (v4).
6. % of rows judged faithful under three (deletion, insertion) threshold configs
   (v4, from the components file).
7. Mean word count and mean model confidence per bucket (v4).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
N_RESAMPLES = 10_000
SEED = 42
BUCKETS = ["low", "medium", "high"]
METRICS = ["lime_faithfulness_score", "shap_faithfulness_score"]
THRESHOLD_CONFIGS = [(0.2, 0.6), (0.3, 0.7), (0.1, 0.5)]

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- load data
v1_adaptive = pd.read_csv(f"{PROCESSED}/faithfulness_results_adaptive.csv")
v4_adaptive = pd.read_csv(f"{PROCESSED}/faithfulness_results_adaptive_v4.csv")
v4_components = pd.read_csv(f"{PROCESSED}/faithfulness_components_v4.csv")
v4_expl = pd.read_csv(f"{PROCESSED}/explanations_output_v4.csv")

sample_frames = []
for b in BUCKETS:
    s = pd.read_csv(f"{PROCESSED}/explainability_sample_{b}.csv")
    s["bucket"] = b
    sample_frames.append(s[["bucket", "text", "effective_score", "n_tokens"]])
samples = pd.concat(sample_frames, ignore_index=True).drop_duplicates(subset=["bucket", "text"])

assert len(v1_adaptive) == len(v4_adaptive) == 900
assert (v1_adaptive["text"].values == v4_adaptive["text"].values).all()
assert (v1_adaptive["bucket"].values == v4_adaptive["bucket"].values).all()

# ---------------------------------------------------------------- bootstrap infra

def cluster_boot_means(df: pd.DataFrame, value_col: str, rng: np.random.Generator, n_resamples: int = N_RESAMPLES) -> np.ndarray:
    """Cluster bootstrap: resample unique texts (with replacement), keeping every
    row of a sampled text. Returns the array of n_resamples bootstrap means."""
    groups = {t: g[value_col].to_numpy() for t, g in df.groupby("text")}
    texts = np.array(list(groups.keys()), dtype=object)
    n_texts = len(texts)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = texts[rng.integers(0, n_texts, size=n_texts)]
        vals = np.concatenate([groups[t] for t in sampled])
        boot_means[i] = vals.mean()
    return boot_means


def ci_from_boot(boot: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(boot, [2.5, 97.5]))


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f}  [{lo:.4f}, {hi:.4f}]"


print("=" * 100)
print("STEP C — ITEM 1: Primary analysis, ALL 900 ROWS, cluster bootstrap (resample unique texts/bucket)")
print("=" * 100)

boot_cache = {}  # (version, bucket, metric) -> boot array

for version, df in (("v1", v1_adaptive), ("v4", v4_adaptive)):
    print(f"\n--- {version} ---")
    print(f"{'bucket':8s} {'metric':6s} {'n_rows':>7s} {'n_texts':>8s}  {'mean [95% CI]'}")
    for bucket in BUCKETS:
        bdf = df[df["bucket"] == bucket]
        for metric, label in zip(METRICS, ["LIME", "SHAP"]):
            boot = cluster_boot_means(bdf, metric, rng)
            boot_cache[(version, bucket, metric)] = boot
            mean = bdf[metric].mean()
            lo, hi = ci_from_boot(boot)
            n_texts = bdf["text"].nunique()
            print(f"{bucket:8s} {label:6s} {len(bdf):7d} {n_texts:8d}  {fmt_ci(mean, lo, hi)}")

print("\n--- Medium - Low and Medium - High (independent-group bootstrap difference) ---")
print(f"{'version':8s} {'metric':6s} {'contrast':14s}  {'diff [95% CI]'}")
for version in ("v1", "v4"):
    for metric, label in zip(METRICS, ["LIME", "SHAP"]):
        med = boot_cache[(version, "medium", metric)]
        low = boot_cache[(version, "low", metric)]
        high = boot_cache[(version, "high", metric)]
        d_ml = med - low
        d_mh = med - high
        print(f"{version:8s} {label:6s} {'Medium-Low':14s}  {fmt_ci(d_ml.mean(), *ci_from_boot(d_ml))}")
        print(f"{version:8s} {label:6s} {'Medium-High':14s}  {fmt_ci(d_mh.mean(), *ci_from_boot(d_mh))}")


print()
print("=" * 100)
print("STEP C — ITEM 2: Sensitivity analysis, UNIQUE (text, bucket) ONLY (duplicates averaged)")
print("=" * 100)

dedup_cache = {}

for version, df in (("v1", v1_adaptive), ("v4", v4_adaptive)):
    dedup = df.groupby(["bucket", "text"], as_index=False)[METRICS].mean()
    dedup_cache[version] = dedup
    print(f"\n--- {version} (deduped) ---")
    print(f"{'bucket':8s} {'metric':6s} {'n(unique texts)':>16s}  {'mean [95% CI]'}")
    boot_dedup = {}
    for bucket in BUCKETS:
        bdf = dedup[dedup["bucket"] == bucket]
        for metric, label in zip(METRICS, ["LIME", "SHAP"]):
            boot = cluster_boot_means(bdf.assign(text=bdf["text"]), metric, rng)  # each text now unique -> plain bootstrap
            boot_dedup[(bucket, metric)] = boot
            mean = bdf[metric].mean()
            lo, hi = ci_from_boot(boot)
            print(f"{bucket:8s} {label:6s} {len(bdf):16d}  {fmt_ci(mean, lo, hi)}")

    print(f"\n  {version} Medium-Low / Medium-High (deduped):")
    for metric, label in zip(METRICS, ["LIME", "SHAP"]):
        med = boot_dedup[("medium", metric)]
        low = boot_dedup[("low", metric)]
        high = boot_dedup[("high", metric)]
        d_ml = med - low
        d_mh = med - high
        print(f"    {label:6s} Medium-Low : {fmt_ci(d_ml.mean(), *ci_from_boot(d_ml))}")
        print(f"    {label:6s} Medium-High: {fmt_ci(d_mh.mean(), *ci_from_boot(d_mh))}")


print()
print("=" * 100)
print("STEP C — ITEM 3: Within-text variability (texts appearing >=3 times) + 13x-comment effect on Low mean")
print("=" * 100)

counts = v4_adaptive.groupby(["bucket", "text"]).size()
repeated = counts[counts >= 3]
print(f"\n(bucket,text) pairs appearing >=3 times: {len(repeated)}")
for (bucket, text), n in repeated.items():
    print(f"\n  bucket={bucket}  n_occurrences={n}")
    print(f"  text: {text[:80]}{'...' if len(text) > 80 else ''}")
    for version, df in (("v1", v1_adaptive), ("v4", v4_adaptive)):
        sub = df[(df["bucket"] == bucket) & (df["text"] == text)]
        for metric, label in zip(METRICS, ["LIME", "SHAP"]):
            vals = sub[metric].to_numpy()
            print(f"    {version} {label:6s}: mean={vals.mean():.4f}  sd={vals.std(ddof=1):.4f}  values={np.round(vals,4).tolist()}")

print("\n--- Low-bucket mean with vs without the 13x comment ---")
for version, df in (("v1", v1_adaptive), ("v4", v4_adaptive)):
    low_all = df[df["bucket"] == "low"]
    if len(repeated) > 0:
        rep_text = repeated.index[0][1]  # the 13x text
        low_excl = low_all[low_all["text"] != rep_text]
    else:
        low_excl = low_all
    print(f"\n  {version}:")
    for metric, label in zip(METRICS, ["LIME", "SHAP"]):
        m_all = low_all[metric].mean()
        m_excl = low_excl[metric].mean()
        print(f"    {label:6s}  with (n={len(low_all)}): {m_all:.4f}   without (n={len(low_excl)}): {m_excl:.4f}   delta={m_all - m_excl:+.4f}")


print()
print("=" * 100)
print("STEP C — ITEM 4: Paired v1 vs v4 on identical 900 rows, mean diff per bucket (cluster bootstrap)")
print("=" * 100)

paired = v1_adaptive[["bucket", "text"]].copy()
for metric in METRICS:
    paired[f"{metric}_v1"] = v1_adaptive[metric].values
    paired[f"{metric}_v4"] = v4_adaptive[metric].values
    paired[f"{metric}_diff"] = paired[f"{metric}_v4"] - paired[f"{metric}_v1"]

print(f"\n{'bucket':8s} {'metric':6s} {'n_rows':>7s}  {'mean diff (v4-v1) [95% CI]'}")
for bucket in BUCKETS:
    bdf = paired[paired["bucket"] == bucket]
    for metric, label in zip(METRICS, ["LIME", "SHAP"]):
        boot = cluster_boot_means(bdf, f"{metric}_diff", rng)
        mean = bdf[f"{metric}_diff"].mean()
        lo, hi = ci_from_boot(boot)
        print(f"{bucket:8s} {label:6s} {len(bdf):7d}  {fmt_ci(mean, lo, hi)}")


print()
print("=" * 100)
print("STEP C — ITEM 5: Correlation between effective_score and faithfulness (v4, all 900 rows)")
print("=" * 100)

v4_with_score = v4_adaptive.merge(samples, on=["bucket", "text"], how="left")
missing = v4_with_score["effective_score"].isna().sum()
print(f"\nrows missing effective_score after join: {missing} / {len(v4_with_score)}")

v4_with_score["combined_faithfulness"] = v4_with_score[METRICS].mean(axis=1)

print(f"\n{'metric':24s} {'Pearson r':>10s} {'p-value':>12s}   {'Spearman rho':>13s} {'p-value':>12s}")
for col, label in [("lime_faithfulness_score", "LIME"), ("shap_faithfulness_score", "SHAP"), ("combined_faithfulness", "Combined (mean)")]:
    x = v4_with_score["effective_score"]
    y = v4_with_score[col]
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"{label:24s} {pr:10.4f} {pp:12.2e}   {sr:13.4f} {sp:12.2e}")


print()
print("=" * 100)
print("STEP C — ITEM 6: %% rows judged faithful under 3 threshold configs, per bucket, v4 (from components file)")
print("=" * 100)

for del_thr, ins_thr in THRESHOLD_CONFIGS:
    print(f"\n--- deletion_threshold={del_thr}, insertion_threshold={ins_thr} ---")
    for method in ("lime", "shap"):
        del_faithful = v4_components[f"{method}_prediction_flipped"] | (v4_components[f"{method}_confidence_drop"] >= del_thr)
        ins_faithful = v4_components[f"{method}_prediction_held"] & (v4_components[f"{method}_confidence_retained"] >= ins_thr)
        faithful = del_faithful & ins_faithful
        tmp = v4_components.assign(_faithful=faithful)
        print(f"  {method.upper():5s}  " + "  ".join(
            f"{b}={tmp[tmp['bucket']==b]['_faithful'].mean()*100:5.1f}%" for b in BUCKETS
        ) + f"   overall={tmp['_faithful'].mean()*100:5.1f}%")


print()
print("=" * 100)
print("STEP C — ITEM 7: Mean word count and mean model confidence per bucket (v4)")
print("=" * 100)

v4_expl["word_count"] = v4_expl["text"].str.split().str.len()
print(f"\n{'bucket':8s} {'n':>5s} {'mean word count':>16s} {'mean confidence':>16s}")
for bucket in BUCKETS:
    bdf = v4_expl[v4_expl["bucket"] == bucket]
    print(f"{bucket:8s} {len(bdf):5d} {bdf['word_count'].mean():16.2f} {bdf['confidence'].mean():16.4f}")

print("\nDONE")
