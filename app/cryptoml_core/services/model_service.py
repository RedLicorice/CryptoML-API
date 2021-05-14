from cryptoml.util.sliding_window import test_windows, predict_day
from cryptoml.util.flattened_classification_report import flattened_classification_report_imbalanced, roc_auc_report
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
from pydantic.error_wrappers import ValidationError
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def create_models_batch(symbol, items):
    print("Model batch: {}".format(symbol, len(items)))
    with ModelRepository() as model_repo:
        models = []
        for d, t, p in items:
            try:
                m = model_repo.find_by_symbol_dataset_target_pipeline(symbol=d.symbol, dataset=d.name, target=t, pipeline=p)
                logging.info("Model exists: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                models.append(m)
            except ValidationError as e:
                logging.info("Model exists and is invalid: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                pass
            except DocumentNotFoundException as e:
                m = Model(
                    symbol=d.symbol,
                    dataset=d.name,
                    target=t,
                    pipeline=p,
                )
                models.append(model_repo.create(m))
                logging.info("Model created: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                pass
        return models


class ModelService:
    def __init__(self):
        self.model_repo: ModelRepository = ModelRepository()

    def create_classification_models(self, query):
        ds = DatasetService()
        models = []
        if query is None:
            query = {
                {"type": "FEATURES", }
            }
        datasets = ds.query(query)
        # All possible combinations
        all_models = {}
        for d in datasets:
            # Get targets for this symbol
            tgt = ds.get_dataset('target', d.symbol)
            if not d.symbol in all_models:
                all_models[d.symbol] = []
            for t, p in itertools.product(tgt.features, PIPELINE_LIST):
                if t in ['price', 'pct']:
                    continue
                all_models[d.symbol].append((d, t, p))
        # Method to process a batch of items
        results = Parallel(n_jobs=-1)(
            delayed(create_models_batch)(symbol, items) for symbol, items in all_models.items())
        return [item for sublist in results for item in sublist]

    def clear_features(self, query=None):
        return self.model_repo.clear_features(query or {})

    def clear_parameters(self, query=None):
        return self.model_repo.clear_parameters(query or {})

    def clear_tests(self, query=None):
        return self.model_repo.clear_tests(query or {})

    def all(self):
        return [m for m in self.model_repo.iterable()]

    def get_model(self, model_id):
        return self.model_repo.get(model_id)

    def get_model(self, pipeline: str, dataset: str, target: str, symbol: str):
        result = self.model_repo.query({"symbol": symbol, "dataset": dataset, "target": target, "pipeline": pipeline})
        if not result:
            return None
        return result[0]

    def get_test(self, pipeline: str, dataset: str, target: str, symbol: str, window: int):
        result = self.model_repo.get_model_test(pipeline, dataset, target, symbol, window)
        if not result:
            return None
        return result[0]

    @staticmethod
    def parse_test_results(test: ModelTest):
        # Re-convert classification results from test to a DataFrame
        results = pd.DataFrame(test.classification_results)
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        results.index = pd.to_datetime(results.index)
        return results

    def get_test_results(self, pipeline: str, dataset: str, target: str, symbol: str, window: int):
        test = self.get_test(pipeline, dataset, target, symbol, window)
        return ModelService.parse_test_results(test)

    def query_models(self, query, projection: Optional[dict] = None):
        return self.model_repo.query(query, projection)

    def create_model_test(self, *, model: Model, split=0.7, step=None, task_key=None, window=None, **kwargs):
        service = DatasetService()
        ds = service.get_dataset(model.dataset, model.symbol)
        splits = service.get_train_test_split_indices(ds, split)
        parameters = kwargs.get('parameters')
        features = kwargs.get('features')
        if isinstance(parameters, str) and parameters == 'latest':
            if model.parameters:
                parameters = model.parameters[-1].parameters
            else:
                parameters = None
        if isinstance(features, str) and features == 'latest':
            if model.features:
                features = model.features[-1].features
            else:
                features = None
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
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]" \
                                   .format(model.pipeline, model.dataset, model.symbol, mt.window))
        X = ds.get_features(model.dataset, model.symbol, begin=begin, end=end)
        y = ds.get_target(model.target, model.symbol, begin=begin, end=end)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(model.symbol, model.dataset, model.target, model.pipeline, unique))
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

        clf_report = flattened_classification_report_imbalanced(df.label, df.predicted)
        roc_report = roc_auc_report(df.label, df.predicted, df[[c for c in df.columns if '_proba_' in c]])
        clf_report.update(roc_report)
        mt.classification_report = clf_report

        self.model_repo.append_test(model.id, mt)

        return mt

    def compare_models(self, symbol: str, dataset: str, target:str):
        tests = self.model_repo.find_tests(symbol=symbol, dataset=dataset, target=target)
        return tests

    def predict_day(self, pipeline: str, dataset: str, target: str, symbol: str, day: str, window: dict):
        model = self.get_model(pipeline, dataset, target, symbol)
        # Load dataset
        ds = DatasetService()
        d = ds.get_dataset(model.dataset, model.symbol)
        # Get training data including the first training window
        begin = sub_interval(timestamp=day, interval=window)
        if from_timestamp(d.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]" \
                                   .format(model.pipeline, model.dataset, model.symbol, window))
        X = ds.get_features(model.dataset, model.symbol, begin=begin, end=day)
        y = ds.get_target(model.target, model.symbol, begin=begin, end=day)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(model.symbol, model.dataset, model.target, model.pipeline, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        # Slice testing interval in windows

        df = predict_day(pipeline_module.estimator, model.parameters[-1], X, y, day)

        return df




