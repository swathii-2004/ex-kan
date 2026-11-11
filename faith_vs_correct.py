"""Does low faithfulness go with wrong predictions?
(Tests the panel's idea instead of assuming it.)

Run from the project root:
    cd ~/Documents/ex-kan
    backend/venv/bin/python3 faith_vs_correct.py

Reads data/processed/faithfulness_results_adaptive_v4.csv (900 rows). It only reads; it changes nothing.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/faithfulness_results_adaptive_v4.csv"
df = pd.read_csv(path)

true_distress = df["label"].isin(["Negative", "Mixed feelings"]).astype(int)
pred_distress = (df["prediction"] == "distress").astype(int)
df["correct"] = (true_distress == pred_distress).astype(int)
df["true_distress"] = true_distress
df["pred_distress"] = pred_distress
df["faith"] = (df["lime_faithfulness_score"] + df["shap_faithfulness_score"]) / 2

print(f"rows: {len(df)} | accuracy on this sample: {df['correct'].mean():.3f}")
print("(optimistic: this sample includes sentences the model trained on)\n")

rho, p = spearmanr(df["faith"], df["correct"])
print(f"Spearman, faithfulness vs correct prediction: rho={rho:+.3f}, p={p:.3f}\n")


def report(bands, title):
    print(title)
    print(f"{'band':<22}{'n':>5}{'accuracy':>10}{'pred distress':>15}{'precision*':>12}")
    for name, sub in bands:
        pdr = sub[sub["pred_distress"] == 1]
        prec = pdr["true_distress"].mean() if len(pdr) else float("nan")
        print(f"{name:<22}{len(sub):>5}{sub['correct'].mean():>10.3f}{len(pdr):>15}{prec:>12.3f}")
    print("*precision = of the sentences predicted distress, the share labelled distress\n")


report([("low  (< 0.40)", df[df.faith < 0.40]),
        ("mid  (0.40 - 0.60)", df[(df.faith >= 0.40) & (df.faith < 0.60)]),
        ("high (>= 0.60)", df[df.faith >= 0.60])], "Fixed bands")

df["tercile"] = pd.qcut(df["faith"], 3, labels=["lowest third", "middle third", "highest third"])
report([(str(k), g) for k, g in df.groupby("tercile", observed=True)], "Terciles of the 900 scores")
