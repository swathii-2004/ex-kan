"""Extra figures and tables for the report, computed from files that already exist. Nothing is retrained or changed.

Part 1  Dataset statistics   (data/processed/low.csv, CLEAN_medium.csv, CLEAN_high.csv, native_script_codemix_*.csv)
Part 2  Faithfulness analyses (data/processed/faithfulness_results_adaptive_v4.csv)

Run from the PROJECT ROOT:
    cd ~/Documents/ex-kan
    backend/venv/bin/python3 extra_analyses.py

It writes 300 dpi figures and CSV tables to the folder report_assets/ and prints every number, so the numbers in the
report can be checked against this output. Paste the printed output to your assistant when asking for the text of the
report sections.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = "data/processed/"
OUT = "report_assets/"
os.makedirs(OUT, exist_ok=True)
TEAL, ORANGE, GREY = "#1F5C52", "#B5541E", "#8FA39E"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})


def save(fig, name):
    fig.savefig(OUT + name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   saved {OUT}{name}")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =============================================================== PART 1: dataset statistics
print("=" * 78 + "\nPART 1: DATASET STATISTICS\n" + "=" * 78)
parts = []
for fname, g in [("low.csv", "Low"), ("CLEAN_medium.csv", "Medium"), ("CLEAN_high.csv", "High"),
                 ("native_script_codemix_medium.csv", "Native script"), ("native_script_codemix_high.csv", "Native script")]:
    p = pd.read_csv(D + fname)
    p["group"] = g
    parts.append(p)
data = pd.concat(parts, ignore_index=True)
data = data[data["label"].isin(["Positive", "Negative", "Mixed feelings"])].copy()
data["words"] = data["text"].astype(str).str.split().str.len()
data["negative_class"] = data["label"].isin(["Negative", "Mixed feelings"])
order = ["Low", "Medium", "High", "Native script"]
print(f"rows: {len(data)}   (expected 4597)")

lab = pd.crosstab(data["group"], data["label"]).reindex(order)[["Positive", "Negative", "Mixed feelings"]]
lab["Total"] = lab.sum(axis=1)
lab["Negative class share"] = ((lab["Negative"] + lab["Mixed feelings"]) / lab["Total"]).round(3)
lab.loc["All"] = [lab["Positive"].sum(), lab["Negative"].sum(), lab["Mixed feelings"].sum(), lab["Total"].sum(),
                  round((lab["Negative"].sum() + lab["Mixed feelings"].sum()) / lab["Total"].sum(), 3)]
print("\nLabels by group:")
print(lab.to_string())
lab.to_csv(OUT + "table_labels_by_group.csv")

ws = data.groupby("group")["words"].agg(["count", "mean", "median", "min", "max"]).reindex(order).round(2)
ws.loc["All"] = [len(data), round(data["words"].mean(), 2), data["words"].median(), data["words"].min(), data["words"].max()]
print("\nWords per comment by group:")
print(ws.to_string())
ws.to_csv(OUT + "table_words_by_group.csv")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True, dpi=300)
for ax, g in zip(axes, ["Low", "Medium", "High"]):
    x = data.loc[data["group"] == g, "words"].clip(upper=40)
    ax.hist(x, bins=range(1, 42, 2), color=TEAL, edgecolor="white")
    ax.set_title(f"{g} (n = {len(x)})", fontsize=9); clean(ax)
    ax.axvline(x.median(), color=ORANGE, ls="--", lw=1); ax.text(x.median() + 1, ax.get_ylim()[1] * 0.85, f"median {x.median():.0f}", color=ORANGE, fontsize=7.5)
axes[0].set_ylabel("Comments"); fig.text(0.5, -0.04, "Words per comment (longer comments counted as 40)", ha="center", fontsize=8.5)
save(fig, "fig_words_hist.png")

fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=300)
sub = lab.loc[["Low", "Medium", "High", "Native script"], ["Positive", "Negative", "Mixed feelings"]]
share = sub.div(sub.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(share))
for col, c in zip(share.columns, [TEAL, ORANGE, GREY]):
    ax.bar(share.index, share[col], bottom=bottom, color=c, label=col, width=0.6)
    for i, v in enumerate(share[col]):
        if v > 6: ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
    bottom += share[col].values
ax.set_ylabel("Share of comments (%)"); ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 1.0)); clean(ax)
save(fig, "fig_labels_by_group.png")

print("\nFive random comments per group (to choose examples for a table; skip anything offensive):")
for g in ["Low", "Medium", "High"]:
    s = data[data["group"] == g].sample(5, random_state=1)
    for _, r in s.iterrows():
        print(f"   [{g}] ({r['label']}) {str(r['text'])[:110]}")

# =============================================================== PART 2: faithfulness analyses
print("\n" + "=" * 78 + "\nPART 2: FAITHFULNESS ANALYSES\n" + "=" * 78)
f = pd.read_csv(D + "faithfulness_results_adaptive_v4.csv")
f["words"] = f["text"].astype(str).str.split().str.len()
f["bucket"] = pd.Categorical(f["bucket"].str.lower(), ["low", "medium", "high"])
f["true_neg"] = f["label"].isin(["Negative", "Mixed feelings"])
f["pred_neg"] = f["prediction"].astype(str).str.strip().str.lower().eq("distress")
f["correct"] = f["true_neg"] == f["pred_neg"]
f["avg"] = (f["lime_faithfulness_score"] + f["shap_faithfulness_score"]) / 2
print(f"sentences: {len(f)}   accuracy on this sample: {f['correct'].mean():.3f}   (the earlier analysis gave 0.862)")

# 1. histograms
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, dpi=300)
for ax, col, name, c in zip(axes, ["lime_faithfulness_score", "shap_faithfulness_score"], ["LIME", "SHAP"], [TEAL, ORANGE]):
    ax.hist(f[col], bins=np.linspace(0, 1, 21), color=c, edgecolor="white"); ax.set_title(name, fontsize=9); clean(ax)
    ax.axvline(f[col].mean(), color="black", ls="--", lw=1); ax.text(f[col].mean() + 0.02, ax.get_ylim()[1] * 0.9, f"mean {f[col].mean():.2f}", fontsize=7.5)
axes[0].set_ylabel("Sentences"); fig.text(0.5, -0.04, "Faithfulness score", ha="center", fontsize=8.5)
save(fig, "fig_faith_hist.png")

# 2. score by number of words (shows the effect of the three-word minimum)
bins = [0, 3, 5, 8, 12, 1000]
labels = ["1 to 3", "4 to 5", "6 to 8", "9 to 12", "13 or more"]
f["len_bin"] = pd.cut(f["words"], bins=bins, labels=labels)
lb = f.groupby("len_bin", observed=True).agg(n=("words", "size"), LIME=("lime_faithfulness_score", "mean"), SHAP=("shap_faithfulness_score", "mean")).round(3)
print("\nMean faithfulness by number of words:")
print(lb.to_string())
lb.to_csv(OUT + "table_faith_by_length_bin.csv")
fig, ax = plt.subplots(figsize=(5.0, 2.9), dpi=300)
x = np.arange(len(lb)); w = 0.36
b1 = ax.bar(x - w / 2, lb["LIME"], w, color=TEAL, label="LIME"); b2 = ax.bar(x + w / 2, lb["SHAP"], w, color=ORANGE, label="SHAP")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n = {n})" for l, n in zip(lb.index, lb["n"])], fontsize=8)
ax.set_ylabel("Mean faithfulness score"); ax.set_xlabel("Number of words in the sentence"); ax.set_ylim(0, 0.9); ax.legend(frameon=False, fontsize=8); clean(ax)
save(fig, "fig_faith_by_length_bin.png")

# 3. reliability bands by group (the bands used in the application: below 0.4, 0.4 to 0.6, 0.6 and above)
f["band"] = pd.cut(f["avg"], [-1, 0.4, 0.6, 2], labels=["Below 0.40", "0.40 to 0.60", "0.60 and above"], right=False)
bd = pd.crosstab(f["bucket"], f["band"], normalize="index").mul(100).round(1)
print("\nShare of sentences in each reliability band (average of LIME and SHAP), % by group:")
print(bd.to_string())
bd.to_csv(OUT + "table_bands_by_group.csv")
fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=300)
bottom = np.zeros(len(bd))
for col, c in zip(bd.columns, [ORANGE, GREY, TEAL]):
    ax.bar([str(i).capitalize() for i in bd.index], bd[col], bottom=bottom, color=c, label=col, width=0.6)
    for i, v in enumerate(bd[col]): ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
    bottom += bd[col].values
ax.set_ylabel("Share of sentences (%)"); ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 1.0)); clean(ax)
save(fig, "fig_bands_by_group.png")

# 4. agreement between the two methods
rho, p = spearmanr(f["lime_faithfulness_score"], f["shap_faithfulness_score"])
print(f"\nSpearman correlation between the LIME and SHAP score of the same sentence: rho = {rho:+.3f} (p = {p:.3g})")
fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=300)
jit = np.random.default_rng(0).normal(0, 0.008, len(f))
ax.scatter(f["lime_faithfulness_score"] + jit, f["shap_faithfulness_score"] + jit, s=7, alpha=0.35, color=TEAL)
ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
ax.set_xlabel("LIME faithfulness score"); ax.set_ylabel("SHAP faithfulness score"); ax.set_title(f"Spearman rho = {rho:+.2f}", fontsize=9); clean(ax)
save(fig, "fig_lime_vs_shap.png")

# 5. by predicted class and by correctness
rows = []
for name, key in [("Predicted Positive", ~f["pred_neg"]), ("Predicted Negative", f["pred_neg"]), ("Prediction correct", f["correct"]), ("Prediction wrong", ~f["correct"])]:
    s = f[key]
    rows.append({"subset": name, "n": len(s), "LIME": round(s["lime_faithfulness_score"].mean(), 3), "SHAP": round(s["shap_faithfulness_score"].mean(), 3)})
sub = pd.DataFrame(rows)
print("\nMean faithfulness by predicted class and by correctness:")
print(sub.to_string(index=False))
sub.to_csv(OUT + "table_faith_by_class_and_correctness.csv", index=False)

acc = f.groupby("bucket", observed=True)["correct"].agg(["size", "mean"]).round(3).rename(columns={"size": "n", "mean": "accuracy on this sample"})
print("\nAccuracy on the 900-sentence sample by group (includes some sentences seen in training):")
print(acc.to_string())
acc.to_csv(OUT + "table_accuracy_sample_by_group.csv")
print("\nDone. Figures and tables are in the folder:", OUT)
