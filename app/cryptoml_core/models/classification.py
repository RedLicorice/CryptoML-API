from typing import Optional, List, Dict, Union
from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel
from cryptoml_core.models.common import TimeInterval
from pydantic import validator


# Hold parameter search results for a Model document
class ModelParameters(BaseModel):
    # Search inputs
    cv_interval: TimeInterval = None  # Begin and end timestamps of data used for parameter search
    cv_splits: Optional[int] = 5  # Number of cv splits to perform
    precision_weights: Optional[dict] = {"0": 1.0, "1": 1.0, "2": 1.0}
    features: Optional[List[str]] = None
    # Search results
    task_key: Optional[str] = None  # For making sure not to run the same task twice
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    parameter_search_method: Optional[str] = None
    parameters: Optional[dict] = None  # Parameters resulting from grid search
    cv_results: Optional[List[dict]] = None  # pd.(cv_results_).to_dict('records')
    result_file: Optional[str] = None  # Dataframe with cross validation results


# Hold feature selection results for a Model document
class ModelFeatures(BaseModel):
    dataset: str
    target: str
    symbol: str
    # Search inputs
    feature_selection_method: str = 'importances'
    search_interval: TimeInterval = None  # Begin and end timestamps of data used for search
    # Search results
    task_key: Optional[str] = None  # For making sure not to run the same task twice
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    features: Optional[List[str]] = None  # Features resulting from feature selection
    feature_importances: Optional[dict] = None  # Features resulting from feature selection


# Hold prediction result for a given day
class DayResult(BaseModel):
    time: str
    predicted: int
    label: int


# Hold model testing results for a Model document
class ModelTest(BaseModel):
    # Test input
    window: dict  # Sliding window width (ie 10 days)
    step: Optional[dict] = {'days': 1}  # Sliding interval, how many days to advance at each step
    parameters: dict  # Parameters to use for building the test model
    features: Optional[List[str]] = [] # Names of the features to use for building the test model (Use all if none)
    test_interval: TimeInterval  # Begin and end timestamps of data used for testing
    # Test results
    task_key: Optional[str] = None  # For making sure not to run the same task twice
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    classification_report: Optional[Union[dict, list]] = {}  # Classification report for this run
    classification_results: Optional[Union[dict, list]] = {}


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
    parameters: Optional[Union[List[ModelParameters], None]] = []
    features: Optional[Union[List[ModelFeatures], None]] = []
    # Model test results
    tests: Optional[Union[List[ModelTest], None]] = []

    @validator('parameters', 'features', 'tests', pre=True, each_item=False, always=True)
    def check_is_list(cls, v):
        if not isinstance(v, list):
            return []
        return v or []

class Estimator(DocumentModel):
    # Simulation day this model was trained for
    day: str
    # Trailing window time interval
    window: dict
    # For dataset indexing
    dataset: str
    target: str
    symbol: str
    pipeline: str
    parameters: str
    features: str
    # File name on S3 for model dump
    filename: str

class TradingModel(DocumentModel):
    # For dataset indexing
    symbol: str  # What symbol are we tuning the model for?
    dataset: str
    target: str
    # What pipeline should be used
    pipeline: str
    # Optimization results
    parameters: Optional[Union[List[ModelParameters], ModelParameters, None]] = []
    features: Optional[Union[List[ModelFeatures], ModelFeatures, None]] = []
    # Model test results
    tests: Optional[Union[List[ModelTest], ModelTest, None]] = []
