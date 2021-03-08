from typing import Optional, List, Dict
from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel


# Encapsulates model training hyperparameters, including:
# - Coordinates for getting training data
# - Symbol/Dataset/Target source names
# - Pipeline and parameters to use for training
# class Hyperparameters(DocumentModel):
#     # For dataset indexing
#     symbol: str  # What symbol are we tuning the model for?
#     dataset: str
#     target: str
#     # What pipeline should be used
#     pipeline: str
#     parameters: Optional[dict] = None
#     features: Optional[List[str]] = []


# Hold prediction result for a given day
class TimeInterval(BaseModel):
    begin: str
    end: str


# Hold parameter search results for a Model document
class ModelParameters(BaseModel):
    parameter_search_method: Optional[str] = None
    parameters: Optional[dict] = None  # Parameters resulting from grid search
    features: Optional[List[str]] = None
    cv_interval: Optional[TimeInterval] = None  # Begin and end timestamps of data used for parameter search
    cv_splits: Optional[int] = 5  # Number of cv splits to perform
    precision_weights: Optional[dict] = {0: 1.0, 1: 1.0, 2: 1.0}
    result_file: Optional[str] = None  # Dataframe with cross validation results


# Hold feature selection results for a Model document
class ModelFeatures(BaseModel):
    feature_selection_method: Optional[str] = 'xgboost_top_k'
    features: Optional[List[str]] = None  # Features resulting from feature selection
    features_cv_interval: Optional[TimeInterval] = None  # Begin and end timestamps of data used for feature optimization


# Hold prediction result for a given day
class DayResult(BaseModel):
    time: str
    predicted: int
    label: int


# Hold model testing results for a Model document
class ModelTest(BaseModel):
    window: dict = None  # Sliding window width (ie 10 days)
    step: Optional[dict] = {'days': 1}  # Sliding interval, how many days to advance at each step
    parameters: dict  # Parameters to use for building the test model
    features: Optional[List[str]] = None # Names of the features to use for building the test model (Use all if none)
    test_interval: TimeInterval  # Begin and end timestamps of data used for testing
    classification_report: Optional[dict] = None  # Classification report for this run
    classification_results: Optional[List[DayResult]] = None


# Identifies a combination of symbol, dataset, target and pipeline
# the base for building a model
class Model(DocumentModel):
    # For dataset indexing
    symbol: str  # What symbol are we tuning the model for?
    dataset: str
    target: str
    # What pipeline should be used
    pipeline: str
    # Optimization results
    parameters: Optional[List[ModelParameters]] = []
    features: Optional[List[ModelFeatures]] = []
    # Model test results
    tests: Optional[List[ModelTest]] = []



