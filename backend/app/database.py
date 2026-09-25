import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

client = MongoClient(MONGODB_URI)


def get_db() -> Database:
    return client[MONGODB_DB_NAME]
