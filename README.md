# Ex-KAN

**Ex-KAN: Evaluating XAI Faithfulness for Stress Detection in Kannada-English Code-Mixed Text**

A research application for detecting stress in Kannada-English code-mixed text and
evaluating the faithfulness of explainable AI (XAI) methods — such as SHAP and LIME —
applied to the resulting model predictions.

## Project structure

```
ex-kan/
├── backend/        FastAPI backend (model serving, explainability, API)
├── frontend/        React + TypeScript app (Vite)
├── data/            Raw and processed datasets, plus data-prep scripts
├── model/           Training code and model checkpoints
└── notebooks/       Exploratory analysis and experiments
```

## Getting started

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
