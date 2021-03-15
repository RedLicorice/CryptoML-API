from cryptoml.util.sliding_window import test_windows
from cryptoml.util.flattened_classification_report import flattened_classification_report
from cryptoml.pipelines import get_pipeline, PIPELINE_LIST
from cryptoml_core.util.timestamp import sub_interval, add_interval, from_timestamp, timestamp_windows
from cryptoml_core.models.classification import Model, ModelTest
from cryptoml_core.repositories.classification_repositories import ModelRepository, DocumentNotFoundException
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException
from cryptoml_core.deps.dask import get_client
import logging
import itertools
from typing import Optional
from uuid import uuid4
from cryptoml_core.util.timestamp import get_timestamp
import numpy as np

class ModelService:
    def __init__(self):
        self.model_repo: ModelRepository = ModelRepository()

    def create_classification_models(self, query):
        ds = DatasetService()
        models = []
        if not query:
            query = {
                {"type": "FEATURES", }
            }
        datasets = ds.query(query)
        for d in datasets:
            # Get targets for this symbol
            tgt = ds.get_dataset('target', d.symbol)
            # For each of the available targets
            for t, p in itertools.product(tgt.features, PIPELINE_LIST):
                # Skip price/pct which are regression targets
                if t in ['price', 'pct']:
                    continue
                # Create a model for each available pipeline
                try:
                    m = self.model_repo.find_by_symbol_dataset_target_pipeline(d.symbol, d.name, t, p)
                    logging.info("Model exists: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                    models.append(m)
                except DocumentNotFoundException as e:
                    m = Model(
                        symbol=d.symbol,
                        dataset=d.name,
                        target=t,
                        pipeline=p,
                    )
                    models.append(self.model_repo.create(m))
                    logging.info("Model created: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                    pass
        return models

    def clear_features(self, query = {}):
        return self.model_repo.clear_features(query)

    def clear_parameters(self, query = {}):
        return self.model_repo.clear_parameters(query)

    def clear_tests(self, query = {}):
        return self.model_repo.clear_tests(query)

    def all(self):
        return [m for m in self.model_repo.iterable()]

    def get_model(self, model_id):
        return self.model_repo.get(model_id)

    def query_models(self, query, projection: Optional[dict] = None):
        return self.model_repo.query(query, projection)

    def create_model_test(self, *, model: Model, split=0.7, step=None, task_key=None, window=None, **kwargs):
        service = DatasetService()
        ds = service.get_dataset(model.dataset, model.symbol)
        splits = service.get_train_test_split_indices(ds, split)
        parameters = kwargs.get('parameters')
        features = kwargs.get('features')
        if isinstance(parameters, str) and parameters == 'latest':
            parameters = model.parameters[-1].parameters
        if isinstance(features, str) and features == 'latest':
            features = model.features[-1].features
        result = ModelTest(
            window=window or {'days': 30},
            step=step or ds.interval,
            parameters=parameters or {},
            features=features or [],
            test_interval=splits['test'],
            task_key=task_key or str(uuid4())
        )
        return result

    def test_model(self, model: Model, mt: ModelTest, **kwargs):
        if not model.id:
            model = self.model_repo.create(model)
        if self.model_repo.exist_test(model.id, mt.task_key):
            logging.info("Model {} test {} already executed!".format(model.id, mt.task_key))
            return mt
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

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        # Slice testing interval in windows

        ranges = timestamp_windows(begin, end, mt.window, mt.step)

        # Connect to Dask
        if not kwargs.get('sync'):
            dask = get_client()
        mt.start_at = get_timestamp()
        df = test_windows(pipeline_module.estimator, mt.parameters, X, y, ranges)
        mt.end_at = get_timestamp()

        mt.classification_results = df.to_dict()
        mt.classification_report = flattened_classification_report(df.label, df.predicted)

        self.model_repo.append_test(model.id, mt)

        return mt
