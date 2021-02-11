from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy_utils import JSONType
from ..database import Base

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime, nullable=False)  # Day the measurement correspond to
    symbol = Column(String(12), nullable=False)  # Symbol ticker
    name = Column(String(64), nullable=False)  # Feature name
    value = Column(Float, nullable=True) # Feature value (We use continuous features so it should be OK)

class FeatureGroup(Base):
    __tablename__ = 'feature_groups'
    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False)  # Dataset (feature group) name
    features = Column(JSONType, nullable=False)  # List of features

class Target(Base):
    __tablename__ = 'targets'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime, nullable=False)  # Day the target correspond to
    symbol = Column(String(12), nullable=False)  # Symbol ticker
    name = Column(String(64), nullable=False)  # Target name
    value = Column(Float, nullable=True) # Feature value (We use continuous features so it should be OK)
