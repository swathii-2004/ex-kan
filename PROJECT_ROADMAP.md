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
- [x] ✅ Literature review — confirm gap is unclaimed (Chowdhury & Bahari, Banerjee et al., etc.)
- [x] ✅ Source and audit base dataset (DravidianCodeMix, ~7,655 rows)
- [x] ✅ Fix code-mix bucketing method (script % + Romanized-Kannada detection)
- [x] ✅ Final bucket counts confirmed: Low 3,696 / Medium 1,482 / High 2,458 (via `improved_bucketer.py`, Romanized-Kannada-aware `effective_score` method)
- [x] ✅ Evaluate and reject unusable extra datasets (DOSA, Kaggle Romanized Reviews, offenseval_dravidian)
- [x] ✅ Obtain secondary depression-labeled dataset (Shwetha & Pushpalatha) — 240 net-new rows
- [x] ✅ Document sentiment-as-distress-proxy justification with real citations
- [x] ✅ Design UI/UX direction and build clickable prototype

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
- [ ] ⏳ Load and finalize bucketed dataset (train/val/test split within each bucket)
- [ ] ⏳ Map sentiment labels → distress / not-distress (Negative + Mixed = distress)
- [ ] ⏳ Fine-tune MuRIL for binary classification
- [ ] ⏳ Evaluate baseline accuracy/F1 — confirm model is "good enough" to be a valid test subject (not the research contribution itself)
- [ ] ⏳ Save trained model checkpoint

### Phase 3 — Explainability Pipeline
- [ ] ⏳ Integrate SHAP on trained model (start with 5–10 samples to validate pipeline)
- [ ] ⏳ Integrate LIME on trained model, compare against SHAP output
- [ ] ⏳ Build word-highlighting output format (for UI display)

### Phase 4 — Faithfulness Testing (the core research step)
- [ ] ⏳ Implement deletion test (remove top words, check prediction shift)
- [ ] ⏳ Implement insertion test (keep only top words, check prediction holds)
- [ ] ⏳ Compute per-sentence faithfulness score
- [ ] ⏳ Run faithfulness tests across ALL samples in Low bucket
- [ ] ⏳ Run faithfulness tests across ALL samples in Medium bucket
- [ ] ⏳ Run faithfulness tests across ALL samples in High bucket
- [ ] ⏳ Aggregate average faithfulness score per bucket
- [ ] ⏳ Statistical comparison across buckets — confirm/deny the core hypothesis
- [ ] ⏳ (Secondary) Validate findings against Shwetha's depression-labeled subset

### Phase 5 — Backend (FastAPI)
- [ ] ⏳ Build `/predict` endpoint (text → distress/not-distress + confidence)
- [ ] ⏳ Build `/explain` endpoint (text → SHAP/LIME highlighted words)
- [ ] ⏳ Build `/faithfulness` endpoint (text → faithfulness score)
- [ ] ⏳ Build `/code-mix-ratio` endpoint (text → Low/Medium/High + score)
- [ ] ⏳ Connect endpoints to MongoDB (save each analysis to History)
- [ ] ⏳ Build `/dashboard-stats` endpoint (aggregate faithfulness per bucket, for Dashboard page)

### Phase 6 — Frontend (React)
- [ ] ⏳ Build Analyze screen (per prototype design)
- [ ] ⏳ Build Result screen (prediction + highlights + gauge)
- [ ] ⏳ Build Dashboard screen (metric cards + bar chart, live from backend)
- [ ] ⏳ Build History screen (table from MongoDB)
- [ ] ⏳ Connect all screens to FastAPI backend
- [ ] ⏳ Polish styling to match approved "Lab Report" design system

### Phase 7 — Integration & Testing
- [ ] ⏳ End-to-end test: type text → see real prediction → real explanation → real faithfulness score
- [ ] ⏳ Verify Dashboard reflects real aggregate data, not placeholder numbers
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
