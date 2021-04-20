from cryptoml_core.models.classification import Model, ModelTest, ModelFeatures, ModelParameters
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException
from cryptoml_core.util.timestamp import get_timestamp
from typing import Union, List


class ModelRepository(DocumentRepository):
    __collection__ = 'models'
    __model__ = Model

    def find_by_symbol_dataset_target_pipeline(self, *, symbol: str = None, dataset: str, target: str, pipeline: str) -> Union[Model,List[Model]]:
        query = {}
        if symbol:
            query['symbol'] = symbol
        if dataset:
            query['dataset'] = dataset
        if target:
            query['target'] = target
        if pipeline:
            query['pipeline'] = pipeline
        if len(query.keys()) == 4:
            document = self.collection.find_one(query)
            if not document:
                raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
            return self.__model__.parse_obj(document)
        else:
            cursor = self.collection.find(query)
            return [self.__model__.parse_obj(document) for document in cursor]

    def find_tests(self, *, symbol, dataset, target):
        cursor = self.collection.aggregate([
            {"$match": {"symbol": symbol, "dataset": dataset, "target": target}},
            {"$unwind": "$tests"},
            {"$unset": ["features", "parameters"]}
        ])
        return [document for document in cursor]

    def create(self, model: Model):
        try:
            _model = self.find_by_symbol_dataset_target_pipeline(symbol=model.symbol, dataset=model.dataset, target=model.target, pipeline=model.pipeline)
            self.update(_model.id, model)
        except DocumentNotFoundException:
            model = super(ModelRepository, self).create(model)
        return model

    def append_test(self, model_id: str, test: ModelTest):
        test_dict = test.dict()
        result = self.collection.update_one(
            {"_id": model_id},
            {'$push': {'tests': test_dict}}
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

    def clear_features(self, query):
        result = self.collection.update_many(
            query,
            {"$set": {"updated": get_timestamp(), "features": []}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return result.modified_count

    def clear_parameters(self, query):
        result = self.collection.update_many(
            query,
            {"$set": {"updated": get_timestamp(), "parameters": []}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return result.modified_count

    def clear_tests(self, query):
        result = self.collection.update_many(
            query,
            {"$set": {"updated": get_timestamp(), "tests": []}}
        )
        if not result.modified_count:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=id)
        return result.modified_count