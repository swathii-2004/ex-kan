from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Query, Response
from pymongo.errors import PyMongoError

from app.database import get_history_collection
from app.models.schemas import HistoryDetail, HistoryItem, HistoryListResponse

router = APIRouter()

_UNAVAILABLE_DETAIL = "History database is unreachable. Please try again later."


def _doc_to_item(doc: dict) -> HistoryItem:
    return HistoryItem(
        id=str(doc["_id"]),
        created_at=doc["created_at"].isoformat(),
        text=doc["text"],
        prediction=doc["prediction"],
        confidence=doc["confidence"],
        code_mix_bucket=doc["code_mix_bucket"],
        effective_score=doc["effective_score"],
        lime_faithfulness_score=doc["lime_faithfulness_score"],
        shap_faithfulness_score=doc["shap_faithfulness_score"],
        model_version=doc["model_version"],
    )


@router.get("/history", response_model=HistoryListResponse)
def list_history(limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0)) -> HistoryListResponse:
    try:
        collection = get_history_collection()
        total = collection.count_documents({})
        cursor = collection.find().sort("created_at", -1).skip(skip).limit(limit)
        items = [_doc_to_item(doc) for doc in cursor]
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    return HistoryListResponse(items=items, total=total)


@router.get("/history/{item_id}", response_model=HistoryDetail)
def get_history_item(item_id: str) -> HistoryDetail:
    try:
        object_id = ObjectId(item_id)
    except InvalidId as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc

    try:
        doc = get_history_collection().find_one({"_id": object_id})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Not found")

    item = _doc_to_item(doc)
    return HistoryDetail(
        **item.model_dump(),
        shap_top_words=doc.get("shap_top_words", []),
        shap_scores=doc.get("shap_scores", []),
        lime_top_words=doc.get("lime_top_words", []),
        lime_scores=doc.get("lime_scores", []),
    )


@router.delete("/history/{item_id}", status_code=204)
def delete_history_item(item_id: str) -> Response:
    try:
        object_id = ObjectId(item_id)
    except InvalidId as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc

    try:
        result = get_history_collection().delete_one({"_id": object_id})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(status_code=204)


@router.delete("/history")
def delete_all_history() -> dict[str, int]:
    try:
        result = get_history_collection().delete_many({})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    return {"deleted": result.deleted_count}
