from typing import Optional, List, Set
from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel
from cryptoml_core.models.common import TimeInterval
from typing import List, Union

class FeatureSelection(BaseModel):
    target: str
    # Search inputs
    method: str = 'importances'
    search_interval: TimeInterval = None  # Begin and end timestamps of data used for search
    # Search results
    task_key: Optional[str] = None  # For making sure not to run the same task twice
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    features: Optional[List[str]] = None  # Features resulting from feature selection
    feature_importances: Optional[dict] = None  # Features resulting from feature selection
    shap_values: Optional[str] = None  # SHAP values for explainability

# Store a list of features included in each dataset,
# and for each currency included min/max index
class Dataset(DocumentModel):
    # Name and symbol are dataset keys
    name: str  # Name of the dataset
    symbol: str  # Ticker name, eg BTC or BTCUSD
    type: str = 'FEATURES'  # Type of the dataset (either FEATURES, TARGET or SOURCE)
    count: int  # Number of entries
    index_min: str  # Timestamp of first record
    index_max: str  # Timestamp of last record
    valid_index_min: str # Timestamp of first valid record (Not including nans)
    valid_index_max: str # Timestamp of last valid record (Not including nans)
    interval: dict  # Timedelta args for sampling interval of the features
    features_path: Optional[str]  # S3 Storage bucket location for features
    features: List[str]  # Name of included columns
    feature_indices: Optional[dict] = None  # First and Last valid index for each feature
    feature_selection: Optional[List[FeatureSelection]] = []