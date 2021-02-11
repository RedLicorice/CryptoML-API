from ..repositories import FeatureRepository
from .storage_service import StorageService
import pandas as pd

class FeatureService:
    def __init__(self):
        self.repo: FeatureRepository = FeatureRepository()
        self.storage: StorageService = StorageService()

    def bootstrap_ohlcv_gdrive(self, zipurl, filename):
        pass
