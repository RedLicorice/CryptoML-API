from cryptoml_core.models.tasks import Task
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException
from pymongo import ASCENDING, DESCENDING
from typing import List


class TaskRepository(DocumentRepository):
    __collection__ = 'tasks'
    __model__ = Task

    def yield_sorted(self) -> List[Task]:
        cursor = self.collection.find().sort([('status', ASCENDING)])
        for document in cursor:
            yield self.__model__.parse_obj(document)
