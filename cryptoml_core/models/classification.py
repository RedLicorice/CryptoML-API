from typing import Optional, List
from datetime import datetime
from cryptoml_core.deps.mongodb.document_model import DocumentModel


# Encapsulates model training hyperparameters, including:
# - Coordinates for getting training data
# - Symbol/Dataset/Target source names
# - Pipeline and parameters to use for training
class Hyperparameters(DocumentModel):
    # For dataset indexing
    symbol: str  # What symbol are we tuning the model for?
    dataset: str
    target: str
    # What pipeline should be used
    pipeline: str
    parameters: Optional[dict] = None
    features: Optional[List[str]] = []




# Extends hyperparameters adding information required for a sliding window classification
# index is the first index of the training set
class SlidingWindowClassification(Hyperparameters):
    index: str  # Index of the last day in the window - formatted timestamp
    train_window: Optional[int] = 30  # How many datapoints to use for training (Will use [index-window, index[)
    test_window: Optional[int] = 1 # How many datapoints to use for testing (Will use [index, index + train[)
    window_interval: Optional[str] = 'days'  # Features granularity, can be an instance of timedelta

# Encapsulated a model test result
class ModelBenchmark(DocumentModel):
    hyperparameters: Hyperparameters #
    report: dict
    train_begin: str
    train_end: str
    test_begin: str
    test_end: str
    window: Optional[int]