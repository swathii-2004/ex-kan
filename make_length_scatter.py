"""Draws the picture for 'Figure: Faithfulness score against sentence length' (LIME and SHAP).

It only reads data/processed/faithfulness_results_adaptive_v4.csv and writes report_assets/fig_faith_vs_length.png.
Run from the PROJECT ROOT:
    cd ~/Documents/ex-kan
    backend/venv/bin/python3 make_length_scatter.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAL, ORANGE, GREY = "#1F5C52", "#B5541E", "#5B6670"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
os.makedirs("report_assets", exist_ok=True)

d = pd.read_csv("data/processed/faithfulness_results_adaptive_v4.csv")
d["words"] = d["text"].astype(str).str.split().str.len()
bins = [0, 3, 5, 8, 12, 1000]
labels = ["1-3", "4-5", "6-8", "9-12", "13+"]
d["bin"] = pd.cut(d["words"], bins=bins, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300, sharey=True)
rng = np.random.default_rng(0)
for ax, col, name, c in zip(axes, ["lime_faithfulness_score", "shap_faithfulness_score"], ["LIME", "SHAP"], [TEAL, ORANGE]):
    x = d["words"].clip(upper=40) + rng.normal(0, 0.12, len(d))
    ax.scatter(x, d[col], s=6, alpha=0.30, color=c, linewidths=0)
    means = d.groupby("bin", observed=True)[col].mean()
    mids = [2, 4.5, 6.5, 10, 20]
    ax.plot(mids, means.values, color="black", lw=1.4, marker="o", ms=3.5, label="mean of the length band")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Words in the sentence (longer ones shown at 40)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 41); ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("Faithfulness score")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig("report_assets/fig_faith_vs_length.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved report_assets/fig_faith_vs_length.png")
