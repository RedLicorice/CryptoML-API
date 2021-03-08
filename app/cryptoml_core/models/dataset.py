from typing import Optional, List, Set
from cryptoml_core.deps.mongodb.document_model import DocumentModel

# Store a list of features included in each dataset,
# and for each currency included min/max index
class Dataset(DocumentModel):
    name: str  # Name of the dataset
    type: Optional[str] = 'FEATURES'  # Type of the dataset (either FEATURES, TARGET or SOURCE)
    ticker: str  # Ticker name, eg BTC or BTCUSD
    count: int  # Number of entries
    index_min: str  # Timestamp of first record
    index_max: str  # Timestamp of last record
    valid_index_min: str # Timestamp of first valid record (Not including nans)
    valid_index_max: str # Timestamp of last valid record (Not including nans)
    interval: dict  # Timedelta args for sampling interval of the features
    features_path: Optional[str]  # S3 Storage bucket location for features
    features: List[str]  # Name of included columns
    feature_indices: Optional[dict] = None # First and Last valid index for each feature
    feature_importances: Optional[dict] = None