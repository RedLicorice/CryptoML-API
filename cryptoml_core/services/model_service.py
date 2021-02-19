import pandas as pd
import logging
# APP Dependencies
from .storage_service import StorageService
from .dataset_service import DatasetService
from ..repositories.classification_repositories import HyperparametersRepository, BenchmarkRepository
# CryptoML Lib Dependencies
from cryptoml.models.grid_search import grid_search
from cryptoml.models.testing import trailing_window_test
from cryptoml.models.training import train_model
from cryptoml.util.flattened_classification_report import flattened_classification_report
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
from cryptoml_core.util.timestamp import to_timestamp
from cryptoml_core.models.classification import Hyperparameters, SlidingWindowClassification, SplitClassification, \
    ModelBenchmark


class ModelService:
    def __init__(self):
        self.storage = StorageService()
        self.parameters_repo = HyperparametersRepository()
        self.benchmark_repo = BenchmarkRepository()

    def __get_tag(self, clf: Hyperparameters):
        return '{}-{}-{}-{}-#{}'.format(clf.pipeline, clf.symbol, clf.dataset, clf.target, clf.id)

    # Perform parameter search
    def grid_search(self, clf: SplitClassification, **kwargs):
        # Load dataset
        ds = DatasetService()
        X_train, X_test, y_train, y_test = ds.get_classification_split(clf)

        # Load pipeline
        pipeline_module = get_pipeline(clf.pipeline)

        # Instantiate splitter and scorer
        splitter = BlockingTimeSeriesSplit(n_splits=5)
        scorer = get_weighted_precision_scorer(weights=kwargs.get('weights'))

        # Perform search
        gscv = grid_search(
            est=pipeline_module.estimator,
            parameters=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
            X_train=X_train,
            y_train=y_train,
            cv=splitter,
            scoring=scorer
        )

        # Update "Classification" request with found hyperparameters,
        #  then save it to MongoDB
        clf.parameters = gscv.best_params_
        _clf = self.parameters_repo.create(clf)

        # Parse grid search results to dataframe
        results_df = pd.DataFrame(gscv.cv_results_)

        # Store grid search results on storage
        self.storage.upload_json_obj(gscv.best_params_, 'grid-search-results', 'parameters-{}.json'\
                                     .format(self.__get_tag(_clf)))
        self.storage.save_df(results_df, 'grid-search-results', 'cv_results-{}.csv'\
                             .format(self.__get_tag(_clf)))

        # Test model parameters on the test set, using different windows
        test_reports = {}
        for _w in [30, 90, 150]:
            test_reports[_w] = self.test_model(clf=clf, W=_w)

        # Return result and Hyperparameters
        return results_df, clf, test_reports

    def test_model(self, clf: SplitClassification, **kwargs):
        # Load dataset
        ds = DatasetService()
        X_train, X_test, y_train, y_test = ds.get_classification_split(clf)

        # Load pipeline
        pipeline_module = get_pipeline(clf.pipeline)

        # Test the model with a sliding window approach, with the first training window starting
        #  in the training set.
        try:
            labels, predictions = trailing_window_test(
                est=pipeline_module.estimator,
                parameters=clf.parameters,
                W=kwargs.get('W', 30),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
        except Exception as e:
            logging.exception(e)
            return None

        # Assemble a Benchmark instance containing results for this run,
        #  then save it into the repository
        benchmark = ModelBenchmark(
            hyperparameters=clf,
            report=flattened_classification_report(y_true=labels, y_pred=predictions),
            window=kwargs.get('W', 30),
            train_begin=to_timestamp(X_train.index.min()),
            train_end=to_timestamp(X_train.index.max()),
            test_begin=to_timestamp(X_test.index.min()),
            test_end=to_timestamp(X_test.index.max())
        )
        self.benchmark_repo.create(benchmark)

        return benchmark

    def train_model_day(self, clf: SlidingWindowClassification, **kwargs):
        # Load Dataset
        ds = DatasetService()
        X_train, _, y_train, _ = ds.get_classification_window(clf)

        # Train a model using the specified parameters
        pipeline_module = get_pipeline(clf.pipeline)
        _est = train_model(
            est=pipeline_module.estimator,
            parameters=clf.parameters,
            X_train=X_train,
            y_train=y_train
        )

        # ToDo: Wrap the model and save it for future use
        return _est

