"""
Pre-specified robustness analysis for the v1-vs-v4 faithfulness comparison. Run once;
report whatever comes out. No other analyses, no tuning based on results.

1. OLS of faithfulness on bucket (Low = reference) + log(word_count), SEs clustered by
   unique text, for each (version, explainer). Then the same model adding model
   confidence. Report Medium/High coefficients with 95% CIs and n.
2. Paired cluster bootstrap (10,000 resamples, seed 42, resample unique texts within
   bucket, same sampled texts used for v1 and v4): the difference in contrasts,
   (Medium-Low)_v4 - (Medium-Low)_v1 and (Medium-High)_v4 - (Medium-High)_v1, for LIME
   and SHAP, with 95% CIs.
3. Within each bucket, Spearman correlation between log(word_count) and faithfulness
   (v4 only).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
N_RESAMPLES = 10_000
SEED = 42
BUCKETS = ["low", "medium", "high"]
METRICS = {"lime_faithfulness_score": "LIME", "shap_faithfulness_score": "SHAP"}

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- load + assemble

def build_dataset(adaptive_path: str, expl_path: str) -> pd.DataFrame:
    adaptive = pd.read_csv(adaptive_path)
    expl = pd.read_csv(expl_path)
    assert len(adaptive) == len(expl)
    assert (adaptive["text"].values == expl["text"].values).all()
    assert (adaptive["bucket"].values == expl["bucket"].values).all()
    df = adaptive.copy()
    df["confidence"] = expl["confidence"].values
    df["word_count"] = df["text"].str.split().str.len()
    df["log_word_count"] = np.log(df["word_count"])
    return df


v1 = build_dataset(f"{PROCESSED}/faithfulness_results_adaptive.csv", f"{PROCESSED}/explanations_output.csv")
v4 = build_dataset(f"{PROCESSED}/faithfulness_results_adaptive_v4.csv", f"{PROCESSED}/explanations_output_v4.csv")
assert (v1["text"].values == v4["text"].values).all() and (v1["bucket"].values == v4["bucket"].values).all()

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


print("=" * 100)
print("ITEM 1: OLS of faithfulness on bucket (Low=ref) + log(word_count), SEs clustered by text")
print("=" * 100)

for version, df in DATASETS.items():
    for metric_col, label in METRICS.items():
        print(f"\n--- {version} {label}  (n={len(df)}, n_clusters={df['text'].nunique()}) ---")

        bucket_dummies = pd.get_dummies(df["bucket"], drop_first=False)
        X_base = pd.DataFrame({
            "const": 1.0,
            "medium": bucket_dummies["medium"].astype(float),
            "high": bucket_dummies["high"].astype(float),
            "log_word_count": df["log_word_count"],
        })
        y = df[metric_col]
        groups = df["text"]

        model_base = sm.OLS(y, X_base).fit(cov_type="cluster", cov_kwds={"groups": groups})
        print("  Model A (bucket + log(word_count)):")
        for coef in ("medium", "high"):
            b = model_base.params[coef]
            lo, hi = model_base.conf_int().loc[coef]
            print(f"    {coef:8s} coef={b:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")

        X_conf = X_base.copy()
        X_conf["confidence"] = df["confidence"]
        model_conf = sm.OLS(y, X_conf).fit(cov_type="cluster", cov_kwds={"groups": groups})
        print("  Model B (+ model confidence):")
        for coef in ("medium", "high"):
            b = model_conf.params[coef]
            lo, hi = model_conf.conf_int().loc[coef]
            print(f"    {coef:8s} coef={b:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")


print()
print("=" * 100)
print("ITEM 2: Paired cluster bootstrap of the v4-vs-v1 change in (Medium-Low) and (Medium-High)")
print("        (same resampled texts used for v1 and v4 in each iteration)")
print("=" * 100)

# v1/v4 share identical (bucket, text) rows in identical order, so we can resample
# text IDENTITIES once per iteration and apply them to both versions' values.
for metric_col, label in METRICS.items():
    print(f"\n--- {label} ---")

    per_bucket_texts = {b: v1.loc[v1["bucket"] == b, "text"].unique() for b in BUCKETS}
    v1_groups = {b: {t: g[metric_col].to_numpy() for t, g in v1[v1["bucket"] == b].groupby("text")} for b in BUCKETS}
    v4_groups = {b: {t: g[metric_col].to_numpy() for t, g in v4[v4["bucket"] == b].groupby("text")} for b in BUCKETS}

    boot_contrast_change = {"Medium-Low": np.empty(N_RESAMPLES), "Medium-High": np.empty(N_RESAMPLES)}

    for i in range(N_RESAMPLES):
        means_v1 = {}
        means_v4 = {}
        for b in BUCKETS:
            texts = per_bucket_texts[b]
            sampled = texts[rng.integers(0, len(texts), size=len(texts))]
            means_v1[b] = np.concatenate([v1_groups[b][t] for t in sampled]).mean()
            means_v4[b] = np.concatenate([v4_groups[b][t] for t in sampled]).mean()

        contrast_v1_ml = means_v1["medium"] - means_v1["low"]
        contrast_v4_ml = means_v4["medium"] - means_v4["low"]
        boot_contrast_change["Medium-Low"][i] = contrast_v4_ml - contrast_v1_ml

        contrast_v1_mh = means_v1["medium"] - means_v1["high"]
        contrast_v4_mh = means_v4["medium"] - means_v4["high"]
        boot_contrast_change["Medium-High"][i] = contrast_v4_mh - contrast_v1_mh

    for contrast_name, boot in boot_contrast_change.items():
        lo, hi = ci(boot)
        print(f"  (({contrast_name})_v4 - ({contrast_name})_v1): {fmt_ci(boot.mean(), lo, hi)}")


print()
print("=" * 100)
print("ITEM 3: Spearman correlation between log(word_count) and faithfulness, per bucket (v4)")
print("=" * 100)

print(f"\n{'bucket':8s} {'metric':6s} {'n':>5s} {'Spearman rho':>13s} {'p-value':>12s}")
for bucket in BUCKETS:
    bdf = v4[v4["bucket"] == bucket]
    for metric_col, label in METRICS.items():
        rho, p = spearmanr(bdf["log_word_count"], bdf[metric_col])
        print(f"{bucket:8s} {label:6s} {len(bdf):5d} {rho:13.4f} {p:12.2e}")

print("\nDONE")
