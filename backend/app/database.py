import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

client = MongoClient(MONGODB_URI)


def get_db() -> Database:
    return client[MONGODB_DB_NAME]


# Separate client for the history feature, with a short server-selection timeout so a
# down/unreachable database fails fast (a few seconds) instead of making /explain or
# /history hang on the driver's default ~30s timeout. Kept independent of `client`
# above so this never changes the behavior of /db-check or any existing endpoint.
HISTORY_SERVER_SELECTION_TIMEOUT_MS = 3000
_history_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=HISTORY_SERVER_SELECTION_TIMEOUT_MS)


def get_history_collection() -> Collection:
    return _history_client[MONGODB_DB_NAME]["analyses"]
