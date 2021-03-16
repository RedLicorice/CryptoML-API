from uuid import uuid4
from pydantic import BaseModel
from typing import List
from cryptoml_core.deps.mongodb import MongoClient, Collection, get_client_and_db
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.exceptions import NotFoundException
from typing import Optional


def get_uuid() -> str:
    """Returns an unique UUID (UUID4)"""
    return str(uuid4())


class DocumentNotFoundException(NotFoundException):
    def __init__(self, collection, identifier):
        self.collection = collection
        self.identifier = identifier
        self.message = "Document not found in collection \"{}\" by \"{}\"".format(collection, identifier)
        super(Exception, self).__init__()


class DocumentRepository:
    __collection__: str = None
    __model__: BaseModel = None
    collection: Collection = None
    client: MongoClient = None

    def __init__(self):
        self.connect()

    def __enter__(self):
        if not self.client:
            self.connect()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.client:
            self.client.close()
        return

    def connect(self):
        self.client, db = get_client_and_db()
        self.collection = db[self.__collection__]

    def get(self, id: str):
        document = self.collection.find_one({"_id": id})
        if not document:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return self.__model__.parse_obj(document)

    def query(self, query: dict, projection: Optional[dict] = None) -> List[BaseModel]:
        cursor = self.collection.find(filter=query, projection=projection)
        return [self.__model__.parse_obj(document) for document in cursor]

    def list(self) -> List[BaseModel]:
        cursor = self.collection.find()
        return [self.__model__.parse_obj(document) for document in cursor]

    def iterable(self) -> List[BaseModel]:
        cursor = self.collection.find()
        for document in cursor:
            yield self.__model__.parse_obj(document)

    def create(self, create: BaseModel):
        document = create.dict()
        document["created"] = document["updated"] = get_timestamp()
        document["_id"] = get_uuid()
        document["id"] = document["_id"]
        result = self.collection.insert_one(document)
        assert result.acknowledged

        return self.get(result.inserted_id)

    def update(self, id: str, update: BaseModel):
        document = update.dict()
        document["updated"] = get_timestamp()

        result = self.collection.update_one({"_id": id}, {"$set": document})
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)

    def touch(self, id):
        result = self.collection.update_one({"_id": id}, {"$set": {"updated": get_timestamp()}})
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)

    def delete(self, id: str):
        result = self.collection.delete_one({"_id": id})
        if not result.deleted_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)