from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from cryptoml_core.deps.config import config
from cryptoml_core.util.ident import get_ident


__all__ = ("get_database")

client = None
database = None
ident = None

def get_collection(collection: str) -> Collection:
    global client, database, ident
    cur_ident = get_ident()
    if not client or not database or ident != cur_ident:
        client = MongoClient(config['database']['mongo']['uri'].get(str))
        database = client[config['database']['mongo']['database'].get(str)]
        ident = cur_ident
    return database[collection]
