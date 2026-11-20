"""Random-word baseline for the v4 faithfulness scores ("faithfulness above chance").

Why: with the project's formula, sentences of 3 words or fewer get a LIME score equal to the model's original confidence
whatever the explanation says (the 3 chosen words are the whole sentence). Choosing words at RANDOM gives exactly the same
score for those sentences, so "actual score minus random score" removes this built-in effect and shows how much better
than chance SHAP and LIME really are.

What it does, using the project's own code (backend/app/services/faithfulness.py) and the served v4 model:
  * for each of the 900 sentences, picks the same number of words N (25% of the words, minimum 3, maximum 10) at random,
    runs the real deletion and insertion tests, and averages the score over several random draws;
  * for the SHAP-style test the random units are word pieces (as in the project), for the LIME-style test whitespace words;
  * compares the actual scores with the random scores by group, and repeats the length-adjusted regression on the
    difference.

It only READS the project files and the model. It writes data/processed/random_baseline_v4.csv.

Run from the PROJECT ROOT, plugged in (about 20 to 40 minutes on a CPU laptop). Stop the backend server first.
    cd ~/Documents/ex-kan
    HF_HUB_OFFLINE=1 backend/venv/bin/python3 random_baseline_v4.py --limit 30 --reps 2     # quick trial (a few minutes)
    HF_HUB_OFFLINE=1 backend/venv/bin/python3 random_baseline_v4.py                         # full run (3 draws per sentence)
"""
import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3, help="random draws per sentence")
ap.add_argument("--limit", type=int, default=0, help="only use this many sentences (quick trial)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

from app.services import explainability as ex                                    # loads the served model (v4)
from app.services.faithfulness import (adaptive_top_n, deletion_test, faithfulness_score,
                                       insertion_test, _shap_segments)
from transformers import AutoTokenizer

if getattr(ex, "MODEL_VERSION", "v4") != "v4":
    sys.exit(f"Active model version is {ex.MODEL_VERSION!r}, not 'v4'. Unset EX_KAN_MODEL_VERSION and retry.")

SRC = ROOT / "data" / "processed" / "faithfulness_results_adaptive_v4.csv"
OUT = ROOT / "data" / "processed" / "random_baseline_v4.csv"
df = pd.read_csv(SRC)
if args.limit:
    df = df.sample(args.limit, random_state=1).reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")


@lru_cache(maxsize=None)
def model(text):
    """The project's PredictFn: text -> (label, confidence of that label). Cached, because the tests re-ask the original."""
    p = np.asarray(ex._predict_proba([text]), dtype=float)[0]
    i = int(np.argmax(p))
    return str(ex.LABELS[i]), float(p[i])


def score(text, words, method):
    tk = tokenizer if method == "shap" else None
    return faithfulness_score(deletion_test(text, words, model, method, tk), insertion_test(text, words, model, method, tk))


rng = np.random.default_rng(args.seed)
rows, t0 = [], time.time()
for k, r in df.iterrows():
    text = str(r["text"])
    words = text.split()
    n = len(words)
    if n == 0:
        continue
    N = adaptive_top_n(n)
    pieces = [s.strip() for s in _shap_segments(text, tokenizer) if s.strip()]
    lime_s, shap_s = [], []
    for _ in range(args.reps):
        lime_s.append(score(text, [words[i] for i in rng.choice(n, size=min(N, n), replace=False)], "lime"))
        if pieces:
            shap_s.append(score(text, [pieces[i] for i in rng.choice(len(pieces), size=min(N, len(pieces)), replace=False)], "shap"))
    rows.append(dict(text=text, bucket=r["bucket"], n_words=n, top_n=N, pred=str(r["prediction"]).strip().lower(),
                     lime=r["lime_faithfulness_score"], shap=r["shap_faithfulness_score"],
                     random_lime=float(np.mean(lime_s)), random_shap=float(np.mean(shap_s)) if shap_s else np.nan))
    if (k + 1) % 25 == 0 or k + 1 == len(df):
        done = k + 1
        eta = (time.time() - t0) / done * (len(df) - done) / 60
        print(f"  {done} / {len(df)} sentences | about {eta:.0f} min left", flush=True)
res = pd.DataFrame(rows)
res["lime_above"] = res["lime"] - res["random_lime"]
res["shap_above"] = res["shap"] - res["random_shap"]
res.to_csv(OUT, index=False)
print("saved", OUT)

# ------------------------------------------------------------------ summaries
import statsmodels.formula.api as smf

res["bucket"] = pd.Categorical(res["bucket"], ["low", "medium", "high"])
res["cluster"] = pd.factorize(res["text"])[0]


def boot_ci(x, n_boot=2000, seed=42):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    g = np.random.default_rng(seed)
    m = np.array([x[g.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


print("\n=== Mean faithfulness: actual, random words, and the difference (95% bootstrap interval) ===")
print(f"{'method':<8}{'group':<9}{'n':>5}{'actual':>8}{'random':>8}   actual - random")
for m in ("lime", "shap"):
    for name, sub in [("all", res)] + [(b, res[res["bucket"] == b]) for b in ["low", "medium", "high"]]:
        a, rnd = sub[m].mean(), sub[f"random_{m}"].mean()
        mean, lo, hi = boot_ci(sub[f"{m}_above"])
        print(f"{m:<8}{name:<9}{len(sub):>5}{a:>8.3f}{rnd:>8.3f}   {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print()

print("=== By predicted class (is the lower faithfulness of Negative predictions a property of the explanation or of the test?) ===")
print(f"{'method':<8}{'class':<20}{'n':>5}{'actual':>8}{'random':>8}   actual - random")
for m in ("lime", "shap"):
    for cls, name in (("not-distress", "predicted Positive"), ("distress", "predicted Negative")):
        sub = res[res["pred"] == cls]
        if len(sub) == 0:
            continue
        mean, lo, hi = boot_ci(sub[f"{m}_above"])
        print(f"{m:<8}{name:<20}{len(sub):>5}{sub[m].mean():>8.3f}{sub['random_' + m].mean():>8.3f}   {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print()

TERM = "C(bucket, Treatment('low'))[T.medium]"
print("=== Regression on the 'actual - random' score: score ~ group + log(words), clustered by identical sentence ===")
print(f"{'filter':<20}{'rows':>5} | {'method':<5} {'Medium effect [95% CI]':<26} {'log-length coef [95% CI]'}")
for name, sub in [("all sentences", res), ("more than 3 words", res[res["n_words"] > 3]), ("more than 5 words", res[res["n_words"] > 5])]:
    for m in ("lime", "shap"):
        sub2 = sub.dropna(subset=[f"{m}_above"])
        fit = smf.ols(f"{m}_above ~ C(bucket, Treatment('low')) + np.log(n_words)", data=sub2).fit(
            cov_type="cluster", cov_kwds={"groups": sub2["cluster"]})
        b, (lo, hi) = fit.params[TERM], fit.conf_int().loc[TERM]
        lc, (llo, lhi) = fit.params["np.log(n_words)"], fit.conf_int().loc["np.log(n_words)"]
        print(f"{name:<20}{len(sub2):>5} | {m:<5} {b:+.3f} [{lo:+.3f}, {hi:+.3f}]      {lc:+.3f} [{llo:+.3f}, {lhi:+.3f}]")
print("\nHow to read it:")
print("  * 'actual - random' clearly above 0 means the method chooses words that matter more than chance.")
print("  * If the Medium effect stays close to 0 and the length coefficient shrinks or loses significance, the earlier length")
print("    result was largely the built-in effect of the test on very short sentences.")
