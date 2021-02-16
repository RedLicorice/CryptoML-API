from ..exceptions import MessageException
from ..services.storage_service import StorageService
from pydantic import BaseModel
from typing import List, Optional

class PersistedModel(BaseModel):
    estimator: any
    selection_method: Optional[str] = 'all'
    features: List[str] = []
    training: TrainingParameters


class ModelRepository:
    def __init__(self):
        self.storage = StorageService()

    def save_model(self, model: PersistedModel):
        return self.storage.upload_pickle_obj(
            model.dict(),
            'persisted-models',
            model.training.get_storage_name()
        )

    def get_model(self, training: TrainingParameters):
        if not self.storage.exist_file('persisted-models', training.get_storage_name()):
            return None
        return self.storage.load_pickled_obj(
            'persisted-models',
            training.get_storage_name()
        )