from fastapi import APIRouter

from app.models.schemas import ExplainResponse, TextRequest
from app.services.explainability import _load_model, explain_lime, explain_shap, predict
from app.services.faithfulness import adaptive_top_n, deletion_test, faithfulness_score, insertion_test

router = APIRouter()

# Number of ranked tokens returned per method. explain_shap/explain_lime already
# return the full list sorted by |importance|; this just caps the API payload/UI to a
# reasonable amount for inline highlighting.
TOP_K = 10


@router.post("/explain", response_model=ExplainResponse)
def explain_endpoint(payload: TextRequest) -> ExplainResponse:
    text = payload.text
    tokenizer, _ = _load_model()

    shap_result = explain_shap(text)
    lime_result = explain_lime(text)

    # Live per-sample faithfulness (not just the aggregate bucket average from
    # /dashboard-stats): reuses the words already computed above, so this only adds a
    # handful of extra model calls (~0.4s total, per PROJECT_ROADMAP.md Phase 4
    # benchmarking) on top of the SHAP/LIME cost that already dominates this endpoint.
    top_n = adaptive_top_n(len(text.split()))
    lime_words = [t for t, _ in lime_result.tokens[:top_n]]
    shap_words = [t for t, _ in shap_result.tokens[:top_n]]

    lime_deletion = deletion_test(text, lime_words, predict, method="lime")
    lime_insertion = insertion_test(text, lime_words, predict, method="lime")
    lime_score = faithfulness_score(lime_deletion, lime_insertion)

    shap_deletion = deletion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
    shap_insertion = insertion_test(text, shap_words, predict, method="shap", tokenizer=tokenizer)
    shap_score = faithfulness_score(shap_deletion, shap_insertion)

    return ExplainResponse(
        shap_top_words=[t for t, _ in shap_result.tokens[:TOP_K]],
        shap_scores=[round(s, 4) for _, s in shap_result.tokens[:TOP_K]],
        lime_top_words=[t for t, _ in lime_result.tokens[:TOP_K]],
        lime_scores=[round(s, 4) for _, s in lime_result.tokens[:TOP_K]],
        lime_faithfulness_score=round(lime_score, 4),
        shap_faithfulness_score=round(shap_score, 4),
    )
