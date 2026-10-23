from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.database import get_history_collection

logger = logging.getLogger(__name__)

# Created lazily on first actual use (save or list), not at import/startup time, so a
# down database never slows down app startup.
_index_created = False


def _ensure_index() -> None:
    global _index_created
    if _index_created:
        return
    try:
        get_history_collection().create_index([("created_at", -1)])
        _index_created = True
    except Exception:
        logger.exception("Failed to create history index on created_at")


def save_analysis(fields: dict[str, Any]) -> str | None:
    """Best-effort save of one analysis to the history collection. Never raises —
    any failure (DB down, auth error, etc.) is logged and reported via the None
    return value, so a broken history store can never break /explain."""
    try:
        _ensure_index()
        doc = {**fields, "created_at": datetime.now(timezone.utc)}
        result = get_history_collection().insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        logger.exception("Failed to save analysis to history")
        return None
