from sqlalchemy import Column, Integer, Float, String, DateTime
from ..database import Base

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    group = Column(String, nullable=False)  # Feature group for this series
    day = Column(DateTime, nullable=False)  # Day these symbols correspond to
    symbol = Column(String, nullable=False)  # Symbol ticker
    name = Column(String, nullable=False)  # Feature name
    value = Column(Float, nullable=True) # Feature value (We use continuous features so it should be OK)

class Target(Base):
    __tablename__ = 'targets'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime, nullable=False)  # Day these symbols correspond to
    symbol = Column(String, nullable=False)  # Symbol ticker
    name = Column(String, nullable=False)  # Feature group for this series
    value = Column(Float, nullable=True) # Feature value (We use continuous features so it should be OK)
