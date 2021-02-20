from typing import Optional, List
from datetime import datetime
from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel

class DatasetSymbol(BaseModel):
    count: int
    index_min: int
    index_max: int

class Dataset(DocumentModel):
    name: str
    features: List[str]
    symbols: List[DatasetSymbol]