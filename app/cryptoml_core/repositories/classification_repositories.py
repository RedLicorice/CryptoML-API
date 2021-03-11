# from cryptoml_core.models.classification import Hyperparameters
from cryptoml_core.models.classification import Model, ModelTest, ModelFeatures, ModelParameters
# from cryptoml_core.models.tuning import GridSearch, ModelTest
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException
from cryptoml_core.util.timestamp import get_timestamp


# class HyperparametersRepository(DocumentRepository):
#     __collection__ = 'hyperparameters'
#     __model__ = Hyperparameters
#
#     def find_by_symbol_dataset_target_pipeline(self, symbol: str, dataset: str, target: str, pipeline: str):
#         query = {"symbol": symbol, "dataset": dataset, "target": target, "pipeline":pipeline}
#         document = self.collection.find_one(query)
#         if not document:
#             raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
#         return self.__model__.parse_obj(document)
#
#     def create(self, model: Hyperparameters):
#         try:
#             _model = self.find_by_symbol_dataset_target_pipeline(model.symbol, model.dataset, model.target, model.pipeline)
#             self.update(_model.id, model)
#         except DocumentNotFoundException:
#             model = super(HyperparametersRepository, self).create(model)
#         return model

# class GridSearchRepository(DocumentRepository):
#     __collection__ = 'grid_search_tasks'
#     __model__ = GridSearch
#
# class ModelTestRepository(DocumentRepository):
#     __collection__ = 'model_test_tasks'
#     __model__ = ModelTest

class ModelRepository(DocumentRepository):
    __collection__ = 'models'
    __model__ = Model

    def find_by_symbol_dataset_target_pipeline(self, symbol: str, dataset: str, target: str, pipeline: str) -> Model:
        query = {"symbol": symbol, "dataset": dataset, "target": target, "pipeline":pipeline}
        document = self.collection.find_one(query)
        if not document:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
        return self.__model__.parse_obj(document)

    def create(self, model: Model):
        try:
            _model = self.find_by_symbol_dataset_target_pipeline(model.symbol, model.dataset, model.target, model.pipeline)
            self.update(_model.id, model)
        except DocumentNotFoundException:
            model = super(ModelRepository, self).create(model)
        return model

    def append_test(self, model_id: str, test: ModelTest):
        result = self.collection.update_one(
            {"_id": model_id},
            {'$push': {'tests': test.dict()}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        self.touch(model_id)

    def append_features(self, model_id: str, features: ModelFeatures):
        result = self.collection.update_one(
            {"_id": model_id},
            {'$push': {'features': features.dict()}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        self.touch(model_id)

    def append_features_query(self, query: dict, features: ModelFeatures):
        result = self.collection.update_many(
            query,
            {'$push': {'features': features.dict()}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return result.modified_count

    def append_parameters(self, model_id: str, parameters: ModelParameters):
        result = self.collection.update_one(
            {"_id": model_id},
            {'$push': {'parameters': parameters.dict()}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        self.touch(model_id)

    def exist_parameters(self, model_id, task_key):
        if not task_key:
            return True
        cursor = self.collection.find_one({'_id': model_id, 'parameters.task_key': task_key})
        return cursor is not None

    def exist_features(self, model_id, task_key):
        if not task_key:
            return True
        cursor = self.collection.find_one({'_id': model_id, 'features.task_key': task_key})
        return cursor is not None

    def exist_test(self, model_id, task_key):
        if not task_key:
            return True
        cursor = self.collection.find_one({'_id': model_id, 'tests.task_key': task_key})
        return cursor is not None

    def get_untested(self):
        cursor = self.collection.find({'parameters': {'$size': 0}})
        return [self.__model__.parse_obj(document) for document in cursor]

    def clear(self, query):
        result = self.collection.update_many(
            query,
            {"$set": {"updated": get_timestamp(), "features": [], "parameters": [], "tasks": []}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return result.modified_count
