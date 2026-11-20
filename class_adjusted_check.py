"""Do the predicted class and the share of the sentence that is tested explain the group and length effects?

Two things in the earlier output suggest this check:
  * Sentences predicted Negative had clearly lower faithfulness than sentences predicted Positive, and the groups
    differ in how many Negative comments they contain (Medium: 51% Negative class, Low: 32%).
  * The faithfulness score falls as sentences get longer up to about 9 words, and then hardly changes. The test uses
    N = 25% of the words, at least 3, so for short sentences N is a large share of the sentence.

This script adds these two factors to the regression, one at a time:
    A: score ~ group + log(words)                                   (the model used in the report so far)
    B: A + predicted class (Negative = 1)
    C: B + share of the sentence tested  (= top_n / words, at most 1)

It only reads data/processed/faithfulness_results_adaptive_v4.csv. Run from the PROJECT ROOT:
    cd ~/Documents/ex-kan
    backend/venv/bin/python3 class_adjusted_check.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/faithfulness_results_adaptive_v4.csv"
d = pd.read_csv(path)
d["words"] = d["text"].astype(str).str.split().str.len()
d["bucket"] = pd.Categorical(d["bucket"].str.lower(), ["low", "medium", "high"])
d["pred_neg"] = d["prediction"].astype(str).str.strip().str.lower().eq("distress").astype(int)
d["share_tested"] = np.minimum(d["top_n"], d["words"]) / d["words"]
d["cluster"] = pd.factorize(d["text"].astype(str))[0]

print(f"sentences: {len(d)}")
print("\nShare predicted Negative, by group:")
print((d.groupby("bucket", observed=True)["pred_neg"].mean() * 100).round(1).astype(str).add(" %").to_string())
print("\nMean faithfulness by group and predicted class (n in brackets):")
for m in ("lime", "shap"):
    t = d.pivot_table(index="bucket", columns="pred_neg", values=f"{m}_faithfulness_score", aggfunc=["mean", "size"], observed=True)
    print(f"  {m.upper()}:")
    for g in t.index:
        print(f"    {g:<7} predicted Positive {t[('mean', 0)][g]:.3f} ({int(t[('size', 0)][g])})   predicted Negative {t[('mean', 1)][g]:.3f} ({int(t[('size', 1)][g])})")
print("\nShare of the sentence that is tested (top_n / words), by number of words:")
bins = pd.cut(d["words"], [0, 3, 5, 8, 12, 1000], labels=["1-3", "4-5", "6-8", "9-12", "13+"])
print(d.groupby(bins, observed=True)["share_tested"].mean().round(2).to_string())

MED = "C(bucket, Treatment('low'))[T.medium]"
HIGH = "C(bucket, Treatment('low'))[T.high]"
base = "C(bucket, Treatment('low')) + np.log(words)"
models = [("A: group + log(words)", base), ("B: A + predicted Negative", base + " + pred_neg"),
          ("C: B + share tested", base + " + pred_neg + share_tested")]


def fmt(fit, name):
    if name not in fit.params:
        return "      -      "
    b = fit.params[name]
    lo, hi = fit.conf_int().loc[name]
    return f"{b:+.3f} [{lo:+.3f},{hi:+.3f}]"


for fname, sub in [("ALL SENTENCES", d), ("SENTENCES WITH MORE THAN 3 WORDS", d[d["words"] > 3])]:
    print("\n" + "=" * 100)
    print(f"{fname}  (n = {len(sub)}); coefficients with 95% intervals, errors clustered by identical sentence")
    print("=" * 100)
    print(f"{'method':<5} {'model':<30}{'Medium effect':<24}{'High effect':<24}{'log(words)':<24}{'pred. Negative':<24}{'share tested'}")
    for m in ("lime", "shap"):
        y = f"{m}_faithfulness_score"
        for label, rhs in models:
            fit = smf.ols(f"{y} ~ {rhs}", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["cluster"]})
            print(f"{m:<5} {label:<30}{fmt(fit, MED):<24}{fmt(fit, HIGH):<24}{fmt(fit, 'np.log(words)'):<24}{fmt(fit, 'pred_neg'):<24}{fmt(fit, 'share_tested')}")
        print()
print("How to read it:")
print("  * Model A should reproduce the earlier Medium effect (+0.003 for LIME, -0.001 for SHAP) when all sentences are used.")
print("  * 'pred. Negative' below 0 (interval not containing 0) means sentences predicted Negative are explained less faithfully.")
print("  * If the log(words) slope shrinks from A to C, part of the length effect is the share of the sentence that is tested.")
print("  * If the Medium effect changes from B or C, class mix or the test design was hiding or creating a group difference.")
print("  * 'share tested' and log(words) are strongly related for short sentences, so treat their separate coefficients with care.")
