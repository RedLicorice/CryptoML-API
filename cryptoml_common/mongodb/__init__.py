from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

# # Package # #
from cryptoml_common.config import config

__all__ = ("client", "database")

client = MongoClient(config['database']['mongo']['uri'].get(str))
database: Database = client[config['database']['mongo']['database'].get(str)]
# collection: Collection = database[config['database']['mongo']['collection'].get(str)]