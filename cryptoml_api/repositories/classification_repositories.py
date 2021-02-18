from ..models.classification import Hyperparameters, ModelBenchmark
from cryptoml_common.mongodb.document_repository import DocumentRepository


class HyperparametersRepository(DocumentRepository):
    __collection__ = 'hyperparameters'
    __model__ = Hyperparameters

class BenchmarkRepository(DocumentRepository):
    __collection__ = 'model_benchmarks'
    __model__ = ModelBenchmark