from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from app.database import get_history_collection

logger = logging.getLogger(__name__)

# Created lazily on first actual use (save or list), not at import/startup time, so a
# down database never slows down app startup.
_index_created = False

# A save does two sequential Mongo round-trips (ensure_index, then insert_one), each
# individually bounded by the client's own timeouts (see database.py) but not bounded
# together — so a slow/unreachable database could still add up to roughly double the
# per-call timeout. Running the whole save in a worker thread with an outer deadline
# gives /explain a single hard ceiling regardless of how many Mongo calls happen
# inside it, or how the driver's own timeouts/retries interact.
SAVE_TIMEOUT_SECONDS = 3.0
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="history-save")


def _ensure_index() -> None:
    global _index_created
    if _index_created:
        return
    try:
        get_history_collection().create_index([("created_at", -1)])
        _index_created = True
    except Exception:
        logger.exception("Failed to create history index on created_at")


def _save_analysis_blocking(fields: dict[str, Any]) -> str | None:
    try:
        _ensure_index()
        doc = {**fields, "created_at": datetime.now(timezone.utc)}
        result = get_history_collection().insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        logger.exception("Failed to save analysis to history")
        return None


def save_analysis(fields: dict[str, Any]) -> str | None:
    """Best-effort save of one analysis to the history collection. Never raises, and
    never delays the caller by more than SAVE_TIMEOUT_SECONDS: if Mongo is slow or
    unreachable, the save keeps running in a background thread (its result is simply
    discarded) while this returns None right away, so a broken history store can
    never meaningfully delay /explain."""
    future = _executor.submit(_save_analysis_blocking, fields)
    try:
        return future.result(timeout=SAVE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning("History save exceeded %.1fs; continuing without saving", SAVE_TIMEOUT_SECONDS)
        return None
