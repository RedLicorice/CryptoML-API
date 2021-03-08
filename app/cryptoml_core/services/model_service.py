import pandas as pd
from cryptoml.models.testing import test_windows
from cryptoml.util.flattened_classification_report import flattened_classification_report
from cryptoml.pipelines import get_pipeline
from cryptoml_core.util.timestamp import sub_interval, add_interval, from_timestamp, timestamp_windows
# from cryptoml_core.models.tuning import ModelTest
# from cryptoml_core.repositories.classification_repositories import ModelTestRepository
from cryptoml_core.models.classification import Model, ModelTest
from cryptoml_core.repositories.classification_repositories import ModelRepository
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException
from cryptoml_core.deps.dask import get_client
import itertools


class ModelService:
    def __init__(self):
        self.model_repo: ModelRepository = ModelRepository()

    def get_model(self, id):
        return self.model_repo.get(id)

    def test_model(self, model: Model, mt: ModelTest, **kwargs):
        if not model.id:
            model = self.model_repo.create(model)
        # Load dataset
        ds = DatasetService()
        d = ds.get_dataset(model.dataset, model.symbol)
        # Get training data including the first training window
        begin = sub_interval(timestamp=mt.test_interval.begin, interval=mt.window)
        end = add_interval(timestamp=mt.test_interval.end, interval=mt.step)
        if from_timestamp(d.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]"\
                                   .format(model.pipeline, model.dataset, model.symbol, mt.window))
        X = ds.get_features(model.dataset, model.symbol, begin=begin, end=end)
        y = ds.get_target(model.target, model.symbol, begin=begin, end=end)

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        # Slice testing interval in windows
        ranges = timestamp_windows(begin, end, mt.window, mt.step)

        # Connect to Dask
        dask = get_client()
        df = test_windows(pipeline_module.estimator, mt.parameters, X, y, ranges)

        mt.classification_results = df.to_dict()
        mt.classification_report = flattened_classification_report(df.label, df.predicted)

        self.model_repo.append_test(model.id, mt)

        return mt
