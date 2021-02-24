from cryptoml_core.models.tasks import Task
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException
from pymongo import ASCENDING, DESCENDING
from typing import List


class TaskRepository(DocumentRepository):
    __collection__ = 'tasks'
    __model__ = Task

    def find_by_task_id(self, task_id: str):
        query = {"task_id": task_id}
        document = self.collection.find_one(query)
        if not document:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
        return self.__model__.parse_obj(document)

    def create(self, model: Task):
        try:
            _model = self.find_by_task_id(model.task_id)
            self.update(_model.id, model)
        except:
            _model = super(TaskRepository, self).create(model)
        return _model

    def yield_sorted(self) -> List[Task]:
        cursor = self.collection.find().sort([('status', ASCENDING)])
        for document in cursor:
            yield self.__model__.parse_obj(document)
