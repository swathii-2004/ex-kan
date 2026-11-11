#!/usr/bin/env python3
"""
Full metrics for the served v4 model on the 460-row test split (Ex-KAN).

Run from the project root:
    cd ~/Documents/ex-kan
    HF_HUB_OFFLINE=1 backend/venv/bin/python3 model/training/metrics_full.py

Check that the script itself works (no model needed):
    backend/venv/bin/python3 model/training/metrics_full.py --self-test

What it does
  1. Rebuilds the exact train/val/test split used for v4 (80/10/10, stratified, seed 42).
  2. Checks that the served model reproduces the recorded test accuracy (0.7522).
  3. Reports, overall and per code-mix group: confusion counts, accuracy, balanced accuracy,
     precision, recall, specificity, F1, macro-F1, weighted-F1, MCC, ROC-AUC, PR-AUC, Brier score,
     with bootstrap 95% confidence intervals for accuracy, F1 and ROC-AUC.
  4. Compares against the "always predict not-distress" baseline.
  5. Saves tables, JSON, per-row predictions, a confusion-matrix plot and ROC curves to
     model/training/results/.
It only READS the project files and the model. It does not change anything else.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                                 matthews_corrcoef, roc_auc_score, roc_curve)
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn is missing. Install it with:  backend/venv/bin/pip install scikit-learn")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "processed"
SEED = 42
SMALL_N = 60
EXPECTED_TEST_ACC = 0.7522          # recorded for v4 (metrics_v4.json)
GROUPS = ["low", "medium", "high", "native_script"]


# ------------------------------------------------------------------ data split (exact v4 recipe)
def build_split():
    files = [("low.csv", "low"), ("CLEAN_medium.csv", "medium"), ("CLEAN_high.csv", "high"),
             ("native_script_codemix_medium.csv", "native_script"),
             ("native_script_codemix_high.csv", "native_script")]
    parts = []
    for fname, bucket in files:
        path = DATA / fname
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = pd.read_csv(path)
        df["bucket"] = bucket
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["label"].isin(["Negative", "Mixed feelings", "Positive"])].copy()
    df["target"] = df["label"].map({"Negative": 1, "Mixed feelings": 1, "Positive": 0})
    df["strat_key"] = df["target"].astype(str) + "_" + df["bucket"]
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["strat_key"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["strat_key"], random_state=SEED)
    return train_df, val_df, test_df


# ------------------------------------------------------------------ metrics
def _safe(num, den):
    return float(num / den) if den else float("nan")


def point_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    acc = _safe(tp + tn, n)
    prec1, rec1 = _safe(tp, tp + fp), _safe(tp, tp + fn)
    prec0, rec0 = _safe(tn, tn + fn), _safe(tn, tn + fp)
    f1_1 = _safe(2 * prec1 * rec1, prec1 + rec1) if not (np.isnan(prec1) or np.isnan(rec1)) else float("nan")
    f1_0 = _safe(2 * prec0 * rec0, prec0 + rec0) if not (np.isnan(prec0) or np.isnan(rec0)) else float("nan")
    both = len(np.unique(y)) == 2
    out = {
        "n": int(n), "n_distress": int(y.sum()), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": acc,
        "balanced_accuracy": float(np.nanmean([rec1, rec0])),
        "precision": prec1, "recall": rec1, "specificity": rec0, "f1": f1_1,
        "macro_f1": float(np.nanmean([f1_1, f1_0])),
        "weighted_f1": float(np.nansum([f1_1 * (tp + fn), f1_0 * (tn + fp)]) / n) if n else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if both and len(np.unique(pred)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
    }
    return out


def bootstrap_ci(y, p, n_boot=2000, thr=0.5, seed=SEED):
    """95% bootstrap intervals for accuracy, F1 (distress class) and ROC-AUC."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        pred = (pb >= thr).astype(int)
        tp = int(((pred == 1) & (yb == 1)).sum()); fp = int(((pred == 1) & (yb == 0)).sum())
        fn = int(((pred == 0) & (yb == 1)).sum())
        accs.append((pred == yb).mean())
        prec, rec = _safe(tp, tp + fp), _safe(tp, tp + fn)
        f1s.append(_safe(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else np.nan)
        aucs.append(roc_auc_score(yb, pb) if len(np.unique(yb)) == 2 else np.nan)
    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2
    return {"accuracy": ci(accs), "f1": ci(f1s), "roc_auc": ci(aucs)}


def full_report(df, n_boot, thr=0.5):
    """df has columns bucket, y_true, p_distress. Returns {group: metrics}."""
    rows = {}
    subsets = [("overall", df)] + [(g, df[df["bucket"] == g]) for g in GROUPS if (df["bucket"] == g).any()]
    for name, sub in subsets:
        m = point_metrics(sub["y_true"], sub["p_distress"], thr)
        ci = bootstrap_ci(sub["y_true"], sub["p_distress"], n_boot, thr)
        for k, (lo, hi) in ci.items():
            m[f"{k}_ci_low"], m[f"{k}_ci_high"] = lo, hi
        m["small_sample"] = bool(len(sub) < SMALL_N)
        # baseline: always predict not-distress
        base = point_metrics(sub["y_true"], np.zeros(len(sub)), thr)
        m["baseline_accuracy"] = base["accuracy"]
        rows[name] = m
    return rows


# ------------------------------------------------------------------ output
def print_table(rows):
    cols = ["n", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "macro_f1",
            "mcc", "roc_auc", "pr_auc", "brier", "baseline_accuracy"]
    t = pd.DataFrame(rows).T[cols]
    for c in cols[1:]:
        t[c] = t[c].astype(float).round(3)
    t["n"] = t["n"].astype(int)
    print("\n=== METRICS (threshold 0.5) ===")
    print(t.to_string())
    print("\n=== BOOTSTRAP 95% INTERVALS (2,000 resamples) ===")
    for g, m in rows.items():
        flag = "  [small sample, interpret with care]" if m["small_sample"] else ""
        print(f"{g:14s} n={m['n']:3d}  acc {m['accuracy']:.3f} [{m['accuracy_ci_low']:.3f}, {m['accuracy_ci_high']:.3f}]"
              f"   F1 {m['f1']:.3f} [{m['f1_ci_low']:.3f}, {m['f1_ci_high']:.3f}]"
              f"   AUC {m['roc_auc']:.3f} [{m['roc_auc_ci_low']:.3f}, {m['roc_auc_ci_high']:.3f}]{flag}")
    o = rows["overall"]
    print(f"\nConfusion matrix (overall): TN={o['TN']}  FP={o['FP']}  FN={o['FN']}  TP={o['TP']}")


def save_outputs(rows, df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "v4_test_predictions.csv", index=False)
    (outdir / "metrics_v4_full.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).T.to_csv(outdir / "metrics_v4_full.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: tables saved, plots skipped "
              "(backend/venv/bin/pip install matplotlib)")
        return
    # confusion matrix (overall), counts and row percentages
    o = rows["overall"]
    cm = np.array([[o["TN"], o["FP"]], [o["FN"], o["TP"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=300)
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / cm[i].sum() if cm[i].sum() else 0
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}% of row)", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Positive", "Negative or mixed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Positive", "Negative or mixed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set, v4)")
    fig.tight_layout(); fig.savefig(outdir / "confusion_matrix_v4.png"); plt.close(fig)
    # ROC curves
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
    for name, sub in [("overall", df)] + [(g, df[df["bucket"] == g]) for g in ["low", "medium", "high"]]:
        if sub["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["y_true"], sub["p_distress"])
        ax.plot(fpr, tpr, lw=2.2 if name == "overall" else 1.3,
                label=f"{name} (AUC {rows[name]['roc_auc']:.2f}, n={rows[name]['n']})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set, v4)"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(outdir / "roc_curves_v4.png"); plt.close(fig)
    print(f"\nSaved to {outdir}")


# ------------------------------------------------------------------ main
def synthetic(n=460, seed=7):
    rng = np.random.default_rng(seed)
    bucket = rng.choice(["low", "medium", "high", "native_script"], size=n, p=[0.56, 0.11, 0.28, 0.05])
    y = (rng.random(n) < 0.36).astype(int)
    p = np.clip(0.5 + (y - 0.5) * 0.45 + rng.normal(0, 0.28, n), 0.01, 0.99)
    return pd.DataFrame({"text": [f"sample {i}" for i in range(n)], "bucket": bucket, "label": "synthetic",
                         "y_true": y, "p_distress": p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="run on synthetic data (no model needed)")
    ap.add_argument("--force", action="store_true", help="continue even if the accuracy check fails")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if args.self_test:
        df = synthetic()
        rows = full_report(df, args.n_boot, args.threshold)
        print_table(rows)
        save_outputs(rows, df, REPO_ROOT / "model" / "training" / "results_selftest")
        print("\nSELF-TEST OK (synthetic data, not real results).")
        return

    _, val_df, test_df = build_split()
    print(f"Split sizes: test={len(test_df)} (expected 460), validation={len(val_df)}")
    if len(test_df) != 460:
        sys.exit("Test size is not 460: the data files differ from the ones used for training. Stop.")

    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.services import explainability as ex          # loads the served model
    if getattr(ex, "MODEL_VERSION", None) != "v4":
        sys.exit(f"Active model version is {ex.MODEL_VERSION!r}, not 'v4'. Unset EX_KAN_MODEL_VERSION and retry.")
    try:
        distress_idx = list(ex.LABELS).index("distress")
    except Exception:
        distress_idx = 1
        print("Could not read LABELS; assuming probability column 1 = distress "
              "(the accuracy check below will catch a wrong guess).")
    texts = test_df["text"].astype(str).tolist()
    print("Scoring 460 test sentences with the v4 model (about 1 to 3 minutes on CPU)...")
    probs = np.asarray(ex._predict_proba(texts), dtype=float)
    p = probs[:, distress_idx] if probs.ndim == 2 else probs

    df = pd.DataFrame({"text": texts, "bucket": test_df["bucket"].values, "label": test_df["label"].values,
                       "y_true": test_df["target"].astype(int).values, "p_distress": p})
    acc = float(((df["p_distress"] >= 0.5).astype(int) == df["y_true"]).mean())
    print(f"Accuracy check: got {acc:.4f}, recorded {EXPECTED_TEST_ACC:.4f}")
    if abs(acc - EXPECTED_TEST_ACC) > 0.005 and not args.force:
        sys.exit("Accuracy does not match the recorded value, so the split or model differs. "
                 "Stop and ask for help (use --force only if you understand why).")

    rows = full_report(df, args.n_boot, args.threshold)
    print_table(rows)
    save_outputs(rows, df, REPO_ROOT / "model" / "training" / "results")


if __name__ == "__main__":
    main()
