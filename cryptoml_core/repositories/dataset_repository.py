from cryptoml_core.models.dataset import Dataset
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException

class DatasetRepository(DocumentRepository):
    __collection__ = 'datasets'
    __model__ = Dataset

    def find_by_dataset_and_symbol(self, dataset: str, symbol: str):
        query = {"name": dataset, "ticker": symbol}
        document = self.collection.find_one(query)
        if not document:
            raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
        return self.__model__.parse_obj(document)

    def yield_by_symbol(self, symbol: str):
        query = {"ticker": symbol}
        cursor = self.collection.find(query)
        if not cursor.count_documents():
            raise DocumentNotFoundException(collection=self.__collection__, identifier=str(query))
        for document in cursor:
            yield self.__model__.parse_obj(document)

    def create(self, model: Dataset):
        try:
            _model = self.find_by_dataset_and_symbol(model.name, model.ticker)
            self.update(_model.id, model)
        except:
            _model = super(DatasetRepository, self).create(model)
        return _model