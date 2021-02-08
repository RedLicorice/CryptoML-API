from ..repositories import FeatureRepository

class FeaturesService:
    def __init__(self):
        self.repo: FeatureRepository = FeatureRepository()
        pass

    def hello(self):
        return "HELLO"