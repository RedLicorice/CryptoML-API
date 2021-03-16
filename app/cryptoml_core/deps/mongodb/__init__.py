from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from cryptoml_core.deps.config import config
from cryptoml_core.util.ident import get_ident
from typing import Tuple, Any


__all__ = ("get_client_and_db", "get_collection")


def get_client_and_db() -> Tuple[MongoClient, Any]:

    c = MongoClient(config['database']['mongo']['uri'].get(str))
    db = c[config['database']['mongo']['database'].get(str)]
    return c, db


def get_collection(collection: str) -> Collection:
    client = MongoClient(config['database']['mongo']['uri'].get(str))
    database = client[config['database']['mongo']['database'].get(str)]
    return database[collection]
