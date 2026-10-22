# Ex-KAN: Evaluating XAI Faithfulness for Stress Detection in Kannada-English Code-Mixed Text

**Student:** Swathi Poojary | MCA, Dept. of CSE, MITE Moodabidri | USN: 4MT25MC103
**Status legend:** ✅ Done · 🔄 In Progress · ⏳ Not Started

---

## 1. Project Description (Product Vision)

### What it is
A research system that tests whether AI explanation tools (SHAP/LIME) can be **trusted** when they explain a stress-detection model's decisions on Kannada-English code-mixed text — and whether that trust **degrades as code-mixing increases**.

### The product experience
A user (researcher, panel, or demo viewer) types or pastes a Kannada-English sentence into a clean, lab-report-styled web app. The app:
1. Detects how "code-mixed" the sentence is (Low / Medium / High)
2. Predicts whether the text signals distress
3. Shows *why* the model made that prediction (highlighted words)
4. Proves whether that explanation is actually trustworthy (faithfulness test)
5. Lets the user see the aggregate research finding: **does trust in explanations fall as mixing increases?**

### Why it matters (the pitch)
Most mental-health AI is tested only on English. Even when a model correctly detects distress in Kannada-English text, we've never verified whether its *explanation* can be trusted — and clinicians, researchers, or platform moderators need trustworthy explanations, not just trustworthy predictions. This project is the first to measure that trust gap for Kannada-English specifically, as a function of mixing intensity, not just a single before/after check.

### Who it's for
- **Primary:** Your research paper / IEEE publication audience
- **Secondary:** A live demo app for your MCA panel/viva to *see* the research in action, not just read about it

---

## 2. Comprehensive Design Explanation

### Design theme: "Lab Report"
Chosen because the whole project is framed as a diagnostic test — SHAP/LIME is the "thermometer," the stress model is the "patient." The UI should feel clinical, credible, and evidence-driven — not playful or consumer-facing.

### Visual language
| Element | Choice | Reason |
|---|---|---|
| Background | Warm paper (#F7F4EC) | Feels like a printed report, not a sterile dashboard |
| Primary accent | Deep teal (#1F5C52) | Trust, clinical calm, used for predictions/system chrome |
| Alert accent | Amber (#B5541E) | Reserved only for distress/high-risk signals — never decorative |
| Display font | Newsreader (serif) | Academic, research-paper feel for headings |
| Body font | Public Sans | Clean, legible, technical |
| Data font | IBM Plex Mono | All numbers/metrics/scores — reinforces "measured data," not opinion |

### Screen-by-screen design intent
1. **Analyze** — minimal friction, one input, one action. Live code-mix badge gives immediate feedback before the user even submits, so they *feel* the system is already "reading" their text.
2. **Result** — the core trust moment. Highlighted words + a semicircular faithfulness gauge given equal visual weight to the prediction itself, because the explanation is the actual subject of study, not a side note.
3. **Dashboard** — this page **is** the research contribution rendered visually. Three metric cards + one bar chart make the central finding ("faithfulness drops as mixing increases") readable in five seconds, for a panel member who has never read the full paper.
4. **History** — supporting evidence trail; shows the system has been tested across many real, varied samples, not cherry-picked demo text.

### Interaction principle
Every screen answers one question: *Analyze* = "what did you say?", *Result* = "can we trust why the model reacted?", *Dashboard* = "does that trust hold up at scale?", *History* = "has this been tested honestly and repeatedly?"

**Prototype link (draft — not finalized, revisit after backend is working):** https://claude.ai/artifact/MgwfNSj752yPvyi8uhNsuN

---

## 3. Full Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language model** | MuRIL / IndicBERT (HuggingFace `transformers`) | Pretrained Kannada-English understanding, fine-tuned for distress classification |
| **Explainability** | SHAP, LIME (Python) | Generate word-level importance explanations |
| **Faithfulness testing** | Custom Python (deletion/insertion logic) | Score whether explanations are trustworthy |
| **Backend** | FastAPI (Python) | Serves `/predict`, `/explain`, `/faithfulness` endpoints |
| **Frontend** | React + TypeScript | 4-screen UI (Analyze / Result / Dashboard / History) |
| **Database** | MongoDB (or PostgreSQL) | Stores analysis history, bucket stats |
| **Data processing** | pandas, scikit-learn | Bucketing, splitting, metrics |
| **Training environment** | PyTorch, HuggingFace `datasets` | Model fine-tuning |
| **Version control** | Git + GitHub | Code + dataset scripts |
| **Dataset** | DravidianCodeMix (7,655 rows, bucketed) + Shwetha & Pushpalatha depression dataset (240 net-new rows, secondary validation) | Training + validation data |

---

## 4. Step-by-Step Development Flow

### Phase 0 — Foundation
- [x] ✅ Define research topic, gap, and question
- [x] ✅ Literature review — confirm gap is unclaimed (Kawser et al. (2025), Banerjee et al., etc.)
- [x] ✅ Source and audit base dataset (DravidianCodeMix, ~7,655 rows)
- [x] ✅ Fix code-mix bucketing method (script % + Romanized-Kannada detection)
- [x] ✅ Final bucket counts confirmed (post contamination-audit, clean counts actually used for training): Low 3,696 / Medium 514 / High 1,280 (via `improved_bucketer.py`, Romanized-Kannada-aware `effective_score` method, then the contamination audit below)
- [x] ✅ Evaluate and reject unusable extra datasets (DOSA, Kaggle Romanized Reviews, offenseval_dravidian)
- [x] ✅ Obtain secondary depression-labeled dataset (Shwetha & Pushpalatha) — 240 net-new rows
- [x] ✅ Document sentiment-as-distress-proxy justification with real citations
- [x] ✅ Design UI/UX direction and build clickable prototype

> High bucket: FULLY audited — all 1,873 rows in the pool (100%), across 4 batches. Final contamination rate: 30.8% (569/1,849 non-native-script rows) — exact population count, not an estimate. Rate remained stable across all 4 batches (27.6%-34.2% range), unlike Medium's upward trend, suggesting High's contamination is a consistent structural property rather than data-ordering artifact. Native-Kannada-script code-mixing (24 rows) tracked separately as valid but out-of-scope. Final clean High count: 1,280 rows — up from the original unaudited 2,458-row bucket.
>
> Medium bucket: FULLY audited — all 1,250 rows in the pool (100%), across 3 batches. Final contamination rate: 50.4% (523/1,036 non-native-script rows). This is an exact population count, not a sample estimate. Contamination rate varied by batch (40.75% → 48.6% → 54.1%), likely due to non-random clustering in the underlying data rather than sampling noise. Native-Kannada-script code-mixing (213 rows) tracked separately as valid but out-of-scope for the Romanized-focused pipeline. Final clean Medium count: 514 rows — nearly 3x the original 176.
>
> Low bucket: spot-checked only (n=100 stratified sample), not fully audited, since monolingual content is expected/correct for this bucket by definition. Result: 95% genuinely monolingual as expected, 5% showed known contamination patterns (2 rows were actual code-mixing misbucketed as Low, 2 were pure-English sentences mentioning "Kannada" as a topic word, 1 weak evidence). Zero Hindi contamination found, unlike Medium/High. This confirms Low is behaving as intended and does not require the same full-audit treatment as Medium/High.

### Phase 1 — Project Setup
- [x] ✅ Create project folder structure (backend/, frontend/, model/, data/, notebooks/)
- [x] ✅ Set up Python virtual environment; requirements.txt created (fastapi, uvicorn, transformers, torch, shap, lime, scikit-learn, pandas, pymongo, python-dotenv)
- [x] ✅ Set up React + TypeScript app scaffold (Vite)
- [x] ✅ Initialize GitHub repo (local commit created)
- [x] ✅ Install full backend dependencies — `pip install -r requirements.txt` (verified in `backend/venv`: torch 2.14, transformers 5.17, shap 0.52, lime 0.2, plus fastapi/uvicorn/pandas/scikit-learn/pymongo/python-dotenv all installed)
- [x] ✅ Push repo to GitHub remote (github.com/swathii-2004/ex-kan) — `origin` configured, local `HEAD` confirmed matching `origin/main`
- [x] ✅ Set up MongoDB instance (Atlas free tier) and connect via `.env` — verified live: `/db-check` returns `{"database":"connected"}` against the real Atlas cluster (fixed by adding IP to Network Access List)
- [ ] ⏳ Prepare Shwetha's 240 net-new rows as a SEPARATE validation file (NOT merged into training data) — tag source clearly, apply label mapping (Depressive→distress, Non-Depressive→not-distress, Neutral→exclude), save as `data/processed/validation_depression_labeled.csv`. Used only AFTER model training, to sanity-check whether the sentiment-proxy-trained model also correctly identifies genuinely depression-labeled text as distress. (Confirmed: no such file exists anywhere in the repo yet.)

### Phase 2 — Model Training
- [x] ✅ Load and finalize bucketed dataset (train/val/test split within each bucket)
- [x] ✅ Map sentiment labels → distress / not-distress (Negative + Mixed = distress)
- [x] ✅ Fine-tune MuRIL for binary classification
- [x] ✅ Evaluate baseline accuracy/F1 — v1 baseline (superseded by v4) (trained on fully-audited dataset: Low 3,696 + Medium 514 + High 1,280 + native-script 237): Overall accuracy=72.2%, precision=0.639, recall=0.556, F1=0.595. Per-bucket: Low acc=0.767/f1=0.577 (n=257), Medium acc=0.608/f1=0.655 (n=51), High acc=0.656/f1=0.551 (n=128). Native-script (informational, n=24): acc=0.833. Two training runs on this dataset produced similar results (F1 0.632 and 0.595), confirming stability.
- [x] ✅ Save trained model checkpoint (`model/checkpoints/ex_kan_model`, confirmed gitignored)

> **v2 model (2026-10-03, SUPERSEDED by v4 — see below):** v2 model (MuRIL+LSTM hybrid, per guide's suggestion referencing Kawser et al. 2025) was briefly the default active model (EX_KAN_MODEL_VERSION=v2), with automatic fallback to v1 if loading fails. v2 results: 77.6% accuracy, F1=0.694 (up from v1's 72.2%/0.595), with the largest gain in the Medium bucket (60.8% → 72.5%). v2 was dropped after discovering its LSTM ran over raw padded sequences with no masking, making its output depend on batch composition (padding length), not just the input text: max change in P(distress) across batch compositions was 0.466, with 1 of 20 test sentences flipping its predicted class. v2's checkpoint and code are kept, untouched, for reference only — it is not used anywhere in the pipeline.
>
> **v4 model (current default, EX_KAN_MODEL_VERSION=v4):** MuRIL+BiLSTM+Linear(256,2), using `pack_padded_sequence`/`pad_packed_sequence` plus attention-mask-weighted mean pooling — a padding-safe redesign of v2's architecture. Same padding-dependence test as above: max change in P(distress) across batch compositions is 0.000025, with 0 of 20 flips — confirms the bug is fixed. Loading failures raise loudly by default (set `EX_KAN_ALLOW_FALLBACK=1` to restore silent fallback to v1). Results (`model/training/metrics_v4.json`): test accuracy 75.2%, F1=0.657. Per-bucket test accuracy: Low 78.6%, Medium 58.8%, High 71.9%, native-script 91.7% (informational). This is the model actually served by `/predict` and `/explain` (the Dashboard reads the v4 faithfulness results file).
>
> **XLM-R base (evaluated, not integrated):** evaluated in Colab as an alternative encoder — mean validation F1 0.706 vs. 0.689 for MuRIL v4, test accuracy 0.774. Not integrated into the backend; kept as a note for future work, not a claim about the deployed model.

### Phase 3 — Explainability Pipeline
- [x] ✅ Integrate SHAP on trained model (start with 5–10 samples to validate pipeline)
- [x] ✅ Integrate LIME on trained model, compare against SHAP output — Explainability pipeline finalized: SHAP (max_evals='auto', capped at 64 above 10 subword tokens) and LIME (whitespace-based split_expression for Kannada-aware tokenization, adaptive num_samples: 150 below 15 words, 300 at/above — thresholds validated via seeded stability testing showing genuine LIME instability on long/lexically-diverse text at reduced sample counts). Combined average: ~21s/sample. Given ~10.5 hour full-dataset runtime estimate on CPU-only hardware, explanations will be run on a stratified sample (not the full 1,794-row corpus) — see next entry for sample size decision.
- [x] ✅ Build word-highlighting output format (for UI display) — `frontend/src/components/HighlightedText.tsx`, wired into `ResultScreen.tsx` (SHAP/LIME side-by-side highlighted text + ranked word-score chips)

### Phase 4 — Faithfulness Testing (the core research step)
- [x] ✅ Faithfulness testing pipeline built and validated (LIME whole-word + SHAP subword support, length-proportional adaptive top-N)
- [x] ✅ Core research finding computed (v4, length-adjusted): raw per-bucket mean faithfulness is Low > High > Medium (LIME 0.643 / 0.566 / 0.617, SHAP 0.504 / 0.449 / 0.481 for Low / Medium / High). After controlling for `log(word_count)` via clustered-SE OLS, the Medium coefficient shrinks to near zero (+0.003 LIME, −0.001 SHAP; both CIs include 0) — the raw Medium deficit is not clearly distinguishable from a length effect. Longer text lowers faithfulness within every bucket. We cannot distinguish code-mixing effects from length effects with this sample.
>
> **Exploratory (word-count-tertile-stratified) follow-up:** 24 comparisons (3 length tertiles × 2 model versions × 2 explainers × 2 contrasts [Medium−Low, Medium−High]), cluster-bootstrap 95% CIs, no multiplicity correction. For v4, in the shortest tertile the Medium gap is statistically significant for LIME (both Medium−Low and Medium−High), but not for SHAP (both intervals include 0); in the mid and long tertiles, none of v4's Medium contrasts are significant. For v1, the pattern is reversed: no significant Medium contrasts in the shortest tertile, but significant gaps appear in the mid and long tertiles (LIME Medium−High in both; SHAP Medium−Low and Medium−High in long texts). This finding is exploratory, not pre-specified, and should not be treated as a replicated result on its own.
- [ ] ⏳ (Secondary) Validate findings against Shwetha's depression-labeled subset

### Phase 5 — Backend (FastAPI)
- [x] ✅ Build `/predict` endpoint (text → distress/not-distress + confidence) — `backend/app/routers/predict.py`
- [x] ✅ Build `/explain` endpoint (text → SHAP/LIME highlighted words, with live faithfulness scores) — `backend/app/routers/explain.py`
- [ ] ⏳ Build `/faithfulness` endpoint (text → faithfulness score) — not a standalone route; faithfulness scores are currently computed inline inside `/explain`'s response (`lime_faithfulness_score`, `shap_faithfulness_score`)
- [ ] ⏳ Build `/code-mix-ratio` endpoint (text → Low/Medium/High + score) — not a standalone route; `code_mix_bucket`/`effective_score` are returned inline from `/predict` (`app/services/code_mix.py`)
- [ ] ⏳ Connect endpoints to MongoDB (save each analysis to History) — not started; `app/database.py` only exposes a `get_db()` helper, nothing calls it yet
- [x] ✅ Build `/dashboard-stats` endpoint (aggregate faithfulness per bucket, for Dashboard page) — `backend/app/routers/dashboard.py`, now reads v4 faithfulness results and returns `model_version`

### Phase 6 — Frontend (React)
- [x] ✅ Build Analyze screen (per prototype design) — `frontend/src/pages/AnalyzeScreen.tsx`
- [x] ✅ Build Result screen (prediction + highlights + gauge) — `frontend/src/pages/ResultScreen.tsx`
- [x] ✅ Build Dashboard screen (metric cards + bar chart, live from backend) — `frontend/src/pages/DashboardScreen.tsx`
- [ ] ⏳ Build History screen (table from MongoDB) — still a placeholder (`HistoryScreen.tsx`), not finished yet
- [ ] ⏳ Connect all screens to FastAPI backend — Analyze, Result, and Dashboard are connected; History is not (depends on the item above)
- [ ] ⏳ Polish styling to match approved "Lab Report" design system

### Phase 7 — Integration & Testing
- [x] ✅ End-to-end test: type text → see real prediction → real explanation → real faithfulness score — `ResultScreen.tsx` calls `/predict` then `/explain`, which returns live SHAP/LIME faithfulness scores per request
- [x] ✅ Verify Dashboard reflects real aggregate data, not placeholder numbers — `/dashboard-stats` now reads real v4 faithfulness results
- [ ] ⏳ Bug fixes, edge cases (empty input, pure English, pure Kannada)

### Phase 8 — Research Paper
- [ ] ⏳ Write Introduction & Gap (mostly ready from earlier research)
- [ ] ⏳ Write Methodology (dataset, bucketing method, model, faithfulness test design, proxy-label justification with citations)
- [ ] ⏳ Write Results (bucket comparison, statistical significance, figures from Dashboard)
- [ ] ⏳ Write Discussion & Limitations (mention proxy labels, Romanized-Kannada/Hindi ambiguity, dataset size caveats)
- [ ] ⏳ Write Conclusion
- [ ] ⏳ Format references in IEEE style
- [ ] ⏳ Identify and submit to target IEEE-indexed conference/journal

---

## 5. How to Use This File
Update the checkboxes (`[ ]` → `[x]`) and status emoji as each step is completed. This file should be your single source of truth for project progress — update it after every work session so nothing gets lost or repeated.
