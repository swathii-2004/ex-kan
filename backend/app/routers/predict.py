from fastapi import APIRouter

from app.models.schemas import PredictResponse, TextRequest
from app.services.code_mix import analyze_code_mix
from app.services.explainability import predict

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: TextRequest) -> PredictResponse:
    prediction, confidence = predict(payload.text)
    code_mix = analyze_code_mix(payload.text)
    return PredictResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        code_mix_bucket=code_mix.bucket,
        effective_score=code_mix.effective_score,
    )
