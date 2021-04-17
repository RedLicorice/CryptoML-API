import pandas as pd
# APP Dependencies
from cryptoml_core.services.storage_service import StorageService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException, NotFoundException
# CryptoML Lib Dependencies
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer, get_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit, SplitException
from cryptoml.util.import_proxy import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
from cryptoml_core.deps.dask import get_client
from cryptoml_core.models.classification import Model, ModelParameters, ModelFeatures
from cryptoml_core.repositories.classification_repositories import ModelRepository
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.util.dict_hash import dict_hash

# SKLearn
from sklearn.utils import parallel_backend
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import NotFittedError
from uuid import uuid4
import logging

from joblib import cpu_count

import math
import numpy as np


class GridSearchService:
    def __init__(self):
        self.storage = StorageService()
        self.model_repo = ModelRepository()
        self.dataset_service = DatasetService()

    def create_parameters_search(self, model: Model, split: float, **kwargs) -> ModelParameters:
        ds = self.dataset_service.get_dataset(model.dataset, model.symbol)
        splits = self.dataset_service.get_train_test_split_indices(ds, split)

        # Features can either be a list of features to use, or a string
        #   If it is a string, and it is "latest", pick the latest
        features = kwargs.get('features')
        if isinstance(features, str) and features == 'latest':
            if model.features:
                features = model.features[-1].features
            else:
                features = None

        # Determine K for K-fold cross validation based on dataset's sample count
        # Train-test split for each fold is 80% train, the lowest training window for accurate results is 30 samples
        # so we need X samples where X is given by the proportion:
        #       30/0.8 = X/1; X= 30/0.8 = 37.5 ~ 40 samples per fold
        X = 40
        k = 5
        # If samples per fold with 5-fold CV are too low, use 3-folds
        if ds.count / k < X:
            k = 3
        # If samples are still too low, raise a value error
        if ds.count / k < X and not kwargs.get("permissive"):
            raise ValueError("Not enough samples to perform cross validation!")

        result = ModelParameters(
            cv_interval=splits['train'],
            cv_splits=k,
            task_key=kwargs.get('task_key', str(uuid4())),
            features=features or None
        )
        return result

    def _get_dataset_and_pipeline(self, model: Model, mp: ModelParameters, **kwargs):
        if not model.id:  # Make sure the task exists
            model = self.model_repo.create(model)
        if self.model_repo.exist_parameters(model.id, mp.task_key):
            logging.info("Model {} Grid search {} already executed!".format(model.id, mp.task_key))
            return mp

        # Load dataset
        X = self.dataset_service.get_features(model.dataset, model.symbol, mp.cv_interval.begin, mp.cv_interval.end,
                                              columns=mp.features)
        y = self.dataset_service.get_target(model.target, model.symbol, mp.cv_interval.begin, mp.cv_interval.end)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(model.symbol, model.dataset, model.target, model.pipeline, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))
        logging.info("Dataset loaded: X {} y {} (unique: {})".format(X.shape, y.shape, unique))
        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        return pipeline_module, X, y

    def grid_search(self, model: Model, mp: ModelParameters, **kwargs) -> ModelParameters:
        pipeline_module, X, y = self._get_dataset_and_pipeline(model, mp)
        tag = "{}-{}-{}-{}-{}" \
            .format(model.symbol, model.dataset, model.target, model.pipeline, dict_hash(mp.parameters))

        # Perform search
        if not kwargs.get('halving'):
            gscv = GridSearchCV(
                estimator=pipeline_module.estimator,
                param_grid=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
                # cv=BlockingTimeSeriesSplit(n_splits=mp.cv_splits),
                cv=StratifiedKFold(n_splits=mp.cv_splits),
                scoring=get_precision_scorer(),
                verbose=kwargs.get("verbose", 0),
                n_jobs=kwargs.get("n_jobs", None),
                refit=False
            )
        else:
            gscv = HalvingGridSearchCV(
                estimator=pipeline_module.estimator,
                param_grid=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
                factor=2,
                cv=BlockingTimeSeriesSplit(n_splits=mp.cv_splits),
                scoring=get_precision_scorer(),
                verbose=kwargs.get("verbose", 0),
                n_jobs=kwargs.get("n_jobs", cpu_count() / 2),
                refit=False,
                random_state=0
            )

        try:
            mp.start_at = get_timestamp()  # Log starting timestamp
            if kwargs.get('sync', False):
                gscv.fit(X, y)
            else:
                # Only run parallel backend if not sync
                dask = get_client()  # Connect to Dask scheduler
                with parallel_backend('dask'):
                    gscv.fit(X, y)
            mp.end_at = get_timestamp()  # Log ending timestamp
        except SplitException as e:
            logging.exception("Model {} splitting yields single-class folds!\n{}".format(tag, e.message))
            return mp  # Fit failed, don't save this.
        except ValueError as e:
            logging.exception("Model {} raised ValueError!\n{}".format(tag, e))
            return mp  # Fit failed, don't save this.

        # Collect results
        results_df = pd.DataFrame(gscv.cv_results_)

        # Update search request with results
        mp.parameter_search_method = 'halving_grid_search' if kwargs.get('halving') else 'gridsearch'
        mp.parameters = gscv.best_params_
        mp.result_file = 'cv_results-{}.csv'.format(tag)

        # Save grid search results on storage
        if kwargs.get('save', True):
            self.storage.upload_json_obj(mp.parameters, 'grid-search-results', 'parameters-{}.json'.format(tag))
            self.storage.save_df(results_df, 'grid-search-results', mp.result_file)
            # Update model with the new results
            self.model_repo.append_parameters(model.id, mp)

        return mp

    def random_search(self, model: Model, mp: ModelParameters, **kwargs) -> ModelParameters:
        pipeline_module, X, y = self._get_dataset_and_pipeline(model, mp)
        tag = "{}-{}-{}-{}-{}" \
            .format(model.symbol, model.dataset, model.target, model.pipeline, dict_hash(mp.parameters))

        rscv = RandomizedSearchCV(
            estimator=pipeline_module.estimator,
            param_distributions=kwargs.get('param_distributions', pipeline_module.PARAMETER_DISTRIBUTION),
            n_iter=kwargs.get('n_iter', 10),
            cv=StratifiedKFold(n_splits=mp.cv_splits),
            scoring=get_precision_scorer(),
            verbose=kwargs.get("verbose", 0),
            n_jobs=kwargs.get("n_jobs", None),
            refit=False,
            random_state=0
        )

        try:
            mp.start_at = get_timestamp()  # Log starting timestamp
            if kwargs.get('sync', False):
                rscv.fit(X, y)
            else:
                # Only run parallel backend if not sync
                dask = get_client()  # Connect to Dask scheduler
                with parallel_backend('dask'):
                    rscv.fit(X, y)
            mp.end_at = get_timestamp()  # Log ending timestamp
        except SplitException as e:
            logging.exception("Model {} splitting yields single-class folds!\n{}".format(tag, e.message))
            return mp  # Fit failed, don't save this.
        except ValueError as e:
            logging.exception("Model {} raised ValueError!\n{}".format(tag, e))
            return mp  # Fit failed, don't save this.

        # Collect results
        results_df = pd.DataFrame(rscv.cv_results_)

        # Update search request with results
        mp.parameter_search_method = 'randomsearch'
        mp.parameters = rscv.best_params_
        mp.result_file = 'cv_results-{}.csv'.format(tag)

        # Save grid search results on storage
        if kwargs.get('save', True):
            self.storage.upload_json_obj(mp.parameters, 'random-search-results', 'parameters-{}.json'.format(tag))
            self.storage.save_df(results_df, 'random-search-results', mp.result_file)
            # Update model with the new results
            self.model_repo.append_parameters(model.id, mp)

        return mp
