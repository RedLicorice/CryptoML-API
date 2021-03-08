from typing import Optional, List
from datetime import datetime
from cryptoml_core.deps.mongodb.document_model import BaseModel
# from .classification import Hyperparameters
from pydantic import BaseModel


# Extends hyperparameters adding information required to reproduce the same task
# class GridSearch(Hyperparameters):
#     cv_begin: str  # Index to begin doing cross validation
#     cv_end: str  # Index to end cross validation
#     cv_splits: Optional[int] = 5  # Number of cv splits to perform
#     precision_weights: Optional[dict] = {0:1.0, 1:1.0, 2:1.0}
#     result_file: Optional[str] = None  # Dataframe with cross validation results
#     feature_importances: Optional[dict] = None  # Dataframe with cross validation results
#

# class DayResult(BaseModel):
#     time: str
#     predicted: int
#     label: int


# class ModelTestBlueprint(BaseModel):
#     windows: Optional[List[dict]] = [{'days': 30}]
#     step: Optional[dict] = {'days':1}
#     test_begin: str  # Timestamp of the day of the first prediction (Using previous W as training data)
#     test_end: str  # Timestamp of the day of the last prediction

# class ModelTest(Hyperparameters):
#     parameters: dict  # Parameters used for this test, no longer optional
#     window: Optional[dict] = {'days': 30}
#     step: Optional[dict] = {'days': 1}
#     test_begin: str  # Timestamp of the day of the first prediction (Using previous W as training data)
#     test_end: str # Timestamp of the day of the last prediction
#     classification_report: Optional[dict] = None
#     classification_results: Optional[List[DayResult]] = None
