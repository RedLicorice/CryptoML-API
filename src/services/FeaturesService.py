from ..repositories import FeatureRepository

class FeaturesService:

    def __init__(self, repository: FeatureRepository):
        self.repo = repository
        pass

    def hello(self):
        return "hello, world!"