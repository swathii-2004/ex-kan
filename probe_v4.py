"""Informal probe: does the served v4 model handle clearly negative and clearly positive sentences?
Run from the project root:  HF_HUB_OFFLINE=1 backend/venv/bin/python3 probe_v4.py
It only reads the model. Edit the sentences so they sound natural to you (you know Kannada better than I do).
This is a small informal check, not a test set.
"""
import sys
import numpy as np

sys.path.insert(0, "backend")
from app.services import explainability as ex

neg = [
    "nange tumba kashta aagide",
    "ivattu tumba bejar aaytu",
    "nange yenu ishta aagtilla",
    "job alli tumba pressure ide, yaarigu helalla",
    "ee movie waste, time waste aaytu",
    "service tumba bad ide",
    "I feel so tired of everything",
    "everything is going wrong today",
    "nange tumba bejar agtide",
    "yaarigu nanna bagge gottilla, tumba ontiyaagi ide",
    "exam bagge tumba bhaya aagtide",
    "ee video tumba bore aagide",
]
pos = [
    "ee haadu tumba chennagide",
    "nange tumba kushi aaytu",
    "super movie guru",
    "ivattu tumba santosha aaytu",
    "tumba olle acting, nange ishta aaytu",
    "best day ever",
    "thanks a lot, tumba help aaytu",
    "I loved this song",
    "ee video nodi nange tumba ishta aaytu",
    "adbhuta haadu, tumba chennagide",
    "great work, tumba kushi aaytu",
    "nange e movie tumba ishta aaytu",
]
sentences = neg + pos
truth = np.array([1] * len(neg) + [0] * len(pos))            # 1 = Negative, 0 = Positive

try:
    idx = list(ex.LABELS).index("distress")                  # 'distress' is the internal name of the Negative class
except Exception:
    idx = 1
probs = np.asarray(ex._predict_proba(sentences), dtype=float)
p_neg = probs[:, idx] if probs.ndim == 2 else probs
pred = (p_neg >= 0.5).astype(int)

for s, t, p in zip(sentences, truth, p_neg):
    flag = "ok   " if int(p >= 0.5) == t else "WRONG"
    print(f"{flag} true={'Neg' if t else 'Pos'}  P(Negative)={p:.2f}  {s}")
print(f"\nnegative sentences found: {pred[truth == 1].mean():.2f} | positive sentences correct: {1 - pred[truth == 0].mean():.2f} | overall: {(pred == truth).mean():.2f}")
print("(informal probe of", len(sentences), "sentences, not a test set)")
