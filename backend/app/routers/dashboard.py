import csv
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter

from app.models.schemas import BucketStats, DashboardStatsResponse

router = APIRouter()

FAITHFULNESS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "faithfulness_results_adaptive.csv"
)


@router.get("/dashboard-stats", response_model=DashboardStatsResponse)
def dashboard_stats() -> DashboardStatsResponse:
    with open(FAITHFULNESS_PATH) as f:
        rows = list(csv.DictReader(f))

    by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"lime": [], "shap": []})
    for r in rows:
        by_bucket[r["bucket"]]["lime"].append(float(r["lime_faithfulness_score"]))
        by_bucket[r["bucket"]]["shap"].append(float(r["shap_faithfulness_score"]))

    buckets = []
    for bucket in ("low", "medium", "high"):
        scores = by_bucket.get(bucket)
        if not scores or not scores["lime"]:
            continue
        buckets.append(
            BucketStats(
                bucket=bucket,
                n=len(scores["lime"]),
                lime_avg=round(sum(scores["lime"]) / len(scores["lime"]), 4),
                shap_avg=round(sum(scores["shap"]) / len(scores["shap"]), 4),
            )
        )

    return DashboardStatsResponse(buckets=buckets)
