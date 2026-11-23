import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")


def _make_client(**kwargs) -> MongoClient | None:
    # MongoClient validates the URI string synchronously at construction time (before
    # any network call), so a malformed MONGODB_URI raises immediately — e.g. a plain
    # ValueError, not a PyMongoError — and would otherwise crash the whole app at
    # import time. Catching it here keeps that failure contained to "Mongo features
    # are unavailable" instead of "the server can't start at all".
    try:
        return MongoClient(MONGODB_URI, **kwargs)
    except Exception:
        logger.exception("Failed to create MongoClient (check MONGODB_URI)")
        return None


# Short timeouts on every phase of a Mongo operation (server selection, TCP connect,
# and the socket read/write itself) so a down/unreachable database fails fast (a few
# seconds) instead of hanging on the driver's much longer defaults (30s / 20s / 20s
# respectively). Applied to both clients below so /db-check fails fast too.
SHORT_TIMEOUT_MS = 3000

client = _make_client(
    serverSelectionTimeoutMS=SHORT_TIMEOUT_MS,
    connectTimeoutMS=SHORT_TIMEOUT_MS,
    socketTimeoutMS=SHORT_TIMEOUT_MS,
)


def get_db() -> Database:
    if client is None:
        raise PyMongoError("MongoDB client failed to initialize (invalid MONGODB_URI)")
    return client[MONGODB_DB_NAME]


# Separate client for the history feature. Kept independent of `client` above (own
# MongoClient instance) so a problem specific to one never affects the other, even
# though they currently share the same short timeouts.
_history_client = _make_client(
    serverSelectionTimeoutMS=SHORT_TIMEOUT_MS,
    connectTimeoutMS=SHORT_TIMEOUT_MS,
    socketTimeoutMS=SHORT_TIMEOUT_MS,
)


def get_history_collection() -> Collection:
    if _history_client is None:
        raise PyMongoError("History MongoDB client failed to initialize (invalid MONGODB_URI)")
    return _history_client[MONGODB_DB_NAME]["analyses"]
