from cryptoml_core.models.dataset import Dataset
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException


class DatasetRepository(DocumentRepository):
    __collection__ = 'datasets'
    __model__ = Dataset

    def find_by_dataset_and_symbol(self, dataset: str, symbol: str):
        query = {"name": dataset, "symbol": symbol}
        document = self.collection.find_one(query)
        if not document:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
        return self.__model__.parse_obj(document)

    def yield_by_name(self, name: str):
        query = {"name": name}
        cursor = self.collection.find(query)
        for document in cursor:
            yield self.__model__.parse_obj(document)

    def yield_by_symbol(self, symbol: str):
        query = {"symbol": symbol}
        cursor = self.collection.find(query)
        for document in cursor:
            yield self.__model__.parse_obj(document)

    def create(self, model: Dataset):
        try:
            _model = self.find_by_dataset_and_symbol(model.name, model.symbol)
            self.update(_model.id, model)
        except:
            _model = super(DatasetRepository, self).create(model)
        return _model

    def yield_by_type(self, type: str):
        query = {"type": type}
        cursor = self.collection.find(query)
        for document in cursor:
            yield self.__model__.parse_obj(document)
