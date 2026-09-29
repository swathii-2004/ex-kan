from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    code_mix_bucket: str
    effective_score: float


class ExplainResponse(BaseModel):
    shap_top_words: list[str]
    shap_scores: list[float]
    lime_top_words: list[str]
    lime_scores: list[float]
    lime_faithfulness_score: float
    shap_faithfulness_score: float


class BucketStats(BaseModel):
    bucket: str
    n: int
    lime_avg: float
    shap_avg: float


class DashboardStatsResponse(BaseModel):
    buckets: list[BucketStats]
