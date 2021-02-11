from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy_utils import JSONType
from ..database import Base

class GridSearchResult(Base):
    __tablename__ = 'grid_search_results'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime)  # Day this model was trained
    symbol = Column(String(64), nullable=False)  # Symbol this model was trained for
    feature_group = Column(String(64), nullable=False)  # Feature group used to train the model
    target = Column(String(64), nullable=False)  # Target used to train the model
    pipeline = Column(String)  # Pipeline used to build the model
    parameters = Column(JSONType, nullable=True) # Feature value (We use continuous features so it should be OK)
    test_report = Column(JSONType, nullable=True) # Classification Report from the trained model on test data
