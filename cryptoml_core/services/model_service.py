import pandas as pd
from cryptoml.models.testing import test_windows_parallel, test_windows
from cryptoml.util.flattened_classification_report import flattened_classification_report
from cryptoml.pipelines import get_pipeline
from cryptoml_core.util.timestamp import sub_interval, add_interval, from_timestamp, timestamp_windows
from cryptoml_core.models.tuning import ModelTest
from cryptoml_core.repositories.classification_repositories import ModelTestRepository
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException


class ModelService:
    def __init__(self):
        self.mt_repo: ModelTestRepository = ModelTestRepository()

    def test_model(self, mt: ModelTest, **kwargs):
        if not mt.id:
            mt = self.mt_repo.create(mt)
        # Load dataset
        ds = DatasetService()
        d = ds.get_dataset(mt.dataset, mt.symbol)
        # Get training data including the first training window
        begin = sub_interval(timestamp=mt.test_begin, interval=mt.window)
        end = add_interval(timestamp=mt.test_end, interval=mt.step)
        if from_timestamp(d.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]".format(mt.pipeline, mt.dataset, mt.symbol, mt.window))
        X = ds.get_features(mt.dataset, mt.symbol, begin=begin, end=end)
        y = ds.get_target(mt.target, mt.symbol, begin=begin, end=end)

        # Load pipeline
        pipeline_module = get_pipeline(mt.pipeline)

        ranges = timestamp_windows(begin, mt.test_end, mt.window, mt.step)
        df = test_windows_parallel(pipeline_module.estimator, mt.parameters, X, y, ranges)
        mt.classification_results = df.to_dict()
        mt.classification_report = flattened_classification_report(df.label, df.predicted)

        self.mt_repo.update(mt.id, mt)
        return mt