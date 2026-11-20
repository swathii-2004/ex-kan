"""Do very short sentences drive the faithfulness findings?

Why this check exists (from reading backend/app/services/faithfulness.py):
  * The test uses the top N words, N = 25% of the words, at least 3, at most 10.
  * For a sentence of 3 words or fewer, the 3 words cover the whole sentence. Deleting them leaves nothing, which the
    code counts as a flipped prediction (deletion component = original confidence), and keeping them leaves the
    whole sentence (insertion component = original confidence). For LIME the score then equals the model's
    original confidence, whatever the explanation says.
  * For a 4-word sentence, 3 words are 75% of the sentence.

This script repeats the group comparison and the length-adjusted regression after dropping the shortest sentences.
It only reads data/processed/faithfulness_results_adaptive_v4.csv and changes nothing.

Run from the PROJECT ROOT (not from the backend folder):
    cd ~/Documents/ex-kan
    backend/venv/bin/python3 short_sentence_check.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/faithfulness_results_adaptive_v4.csv"
d = pd.read_csv(path)
d["n"] = d["text"].astype(str).str.split().str.len()
d["bucket"] = pd.Categorical(d["bucket"], ["low", "medium", "high"])
d["cluster"] = pd.factorize(d["text"].astype(str))[0]            # identical sentences form one cluster

print(f"rows: {len(d)}")
print("\nSentences by length and group:")
tab = pd.DataFrame({
    "3 words or fewer": (d["n"] <= 3).groupby(d["bucket"], observed=True).sum(),
    "4 words or fewer": (d["n"] <= 4).groupby(d["bucket"], observed=True).sum(),
    "all": d.groupby("bucket", observed=True).size(),
})
print(tab.to_string())

TERM = "C(bucket, Treatment('low'))[T.medium]"
filters = [("all sentences", d), ("more than 3 words", d[d["n"] > 3]),
           ("more than 4 words", d[d["n"] > 4]), ("more than 5 words", d[d["n"] > 5])]

print("\nMean faithfulness by group, and the Medium-group effect after adjusting for log sentence length")
print("(model: score ~ group + log(words), errors clustered by identical sentence; reference group = Low)\n")
header = f"{'filter':<20}{'rows':>5} | {'method':<5} {'Low':>6} {'Med':>6} {'High':>6} | {'Medium effect [95% CI]':<26} {'log-length coef'}"
print(header)
print("-" * len(header))
for name, sub in filters:
    sub = sub.copy()
    for m in ("lime", "shap"):
        y = f"{m}_faithfulness_score"
        means = sub.groupby("bucket", observed=True)[y].mean()
        fit = smf.ols(f"{y} ~ C(bucket, Treatment('low')) + np.log(n)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["cluster"]})
        b = fit.params[TERM]
        lo, hi = fit.conf_int().loc[TERM]
        lc = fit.params["np.log(n)"]
        print(f"{name:<20}{len(sub):>5} | {m:<5} {means['low']:>6.3f} {means['medium']:>6.3f} {means['high']:>6.3f} | "
              f"{b:+.3f} [{lo:+.3f}, {hi:+.3f}]      {lc:+.3f}")
    print()
print("How to read it:")
print("  * The 'all sentences' rows should reproduce your Chapter 4 values (Medium effect about +0.003 for LIME and -0.001 for SHAP).")
print("  * If the Medium effect stays close to zero in every filter, the main finding does not depend on the shortest sentences.")
print("  * If the log-length coefficient becomes much smaller after dropping short sentences, part of the length effect")
print("    comes from the test design (the minimum of 3 words), and the report should say so.")
