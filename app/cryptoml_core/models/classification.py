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
