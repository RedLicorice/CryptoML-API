import pandas as pd
# APP Dependencies
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.storage_service import StorageService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException, NotFoundException
# CryptoML Lib Dependencies
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer, get_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit, SplitException
from cryptoml.util.import_proxy import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
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
        self.model_service = ModelService()
        self.dataset_service = DatasetService()

    def create_parameters_search(self, model: Model, split: float, **kwargs) -> ModelParameters:
        ds = self.dataset_service.get_dataset(model.dataset, model.symbol)
        splits = DatasetService.get_train_test_split_indices(ds, split)

        # Features can either be a list of features to use, or a string
        #   If it is a string, and it is "latest", pick the latest
        features = kwargs.get('features')
        # if isinstance(features, str) and features == 'latest':
        #     if model.features:
        #         features = model.features[-1].features
        #     else:
        #         features = None
        if features:
            target = kwargs.get('target', 'class')
            mf = DatasetService.get_feature_selection(
                ds=ds,
                method=kwargs.get('features'),
                target=target
            )
            if not mf:
                raise MessageException(f"Feature selection not found for {model.dataset}.{model.symbol} -> {target}!")
            features = mf.features

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
        mp.cv_results = results_df.to_dict()
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

    def grid_search_new(self, symbol: str, dataset: str, target: str, pipeline: str, split: float, feature_selection_method: str, **kwargs):
        # Check if a model exists and has same search method
        existing_model = self.model_service.get_model(pipeline=pipeline, dataset=dataset, target=target, symbol=symbol)
        if existing_model:
            mp = ModelService.get_model_parameters(existing_model, method='gridsearch')
            if mp:
                logging.info(f"Grid search already performed for {pipeline}({dataset}.{symbol}) -> {target}")
                return mp

        # Retrieve dataset to use
        ds = self.dataset_service.get_dataset(dataset, symbol)

        # Determine cv_splits=K for K-fold cross validation based on dataset's sample count
        # Train-test split for each fold is 80% train, the lowest training window for accurate results is 30 samples
        # so we need X samples where X is given by the proportion:
        #       30/0.8 = X/1; X= 30/0.8 = 37.5 ~ 40 samples per fold
        X = 40
        cv_splits = 5
        # If samples per fold with 5-fold CV are too low, use 3-folds
        if ds.count / cv_splits < X:
            cv_splits = 3
        # If samples are still too low, raise a value error
        if ds.count / cv_splits < X and not kwargs.get("permissive"):
            raise ValueError("Not enough samples to perform cross validation!")

        # Determine split indices based on dataset
        splits = DatasetService.get_train_test_split_indices(ds, split)
        cv_interval = splits['train']

        # Load dataset features by applying a specified feature selection method
        X = self.dataset_service.get_dataset_features(
            ds=ds,
            begin=cv_interval['begin'],
            end=cv_interval['end'],
            method=feature_selection_method,
            target=target
        )
        y = self.dataset_service.get_target(
            name=target,
            symbol=symbol,
            begin=cv_interval['begin'],
            end=cv_interval['end'],
        )

        # Check number of samples for each class in training data, if less than 3 instances are present for
        # each class, we're going to get a very unstable model (or no model at all for k-NN based algos)
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(symbol, dataset, target, pipeline, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))
        logging.info("Dataset loaded: X {} y {} (unique: {})".format(X.shape, y.shape, unique))

        # Load pipeline algorithm and parameter grid
        pipeline_module = get_pipeline(pipeline)

        # Perform search
        gscv = GridSearchCV(
            estimator=pipeline_module.estimator,
            param_grid=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
            # cv=BlockingTimeSeriesSplit(n_splits=mp.cv_splits),
            cv=StratifiedKFold(n_splits=cv_splits),
            scoring=get_precision_scorer(),
            verbose=kwargs.get("verbose", 0),
            n_jobs=kwargs.get("n_jobs", None),
            refit=False
        )

        mp = ModelParameters(
            cv_interval=splits['train'],
            cv_splits=cv_splits,
            task_key=kwargs.get('task_key', str(uuid4())),
            features=[c for c in X.columns],
            parameter_search_method='gridsearch'
        )

        mp.start_at = get_timestamp()
        gscv.fit(X, y)
        mp.end_at = get_timestamp()

        # Collect results
        results_df = pd.DataFrame(gscv.cv_results_)

        mp.parameters = gscv.best_params_
        mp.cv_results = results_df.loc[:, results_df.columns != 'params'].to_dict('records')

        tag = "{}-{}-{}-{}-{}".format(
            symbol,
            dataset,
            target,
            pipeline,
            dict_hash(mp.parameters)
        )
        mp.result_file = 'cv_results-{}.csv'.format(tag)

        # Is there an existing model for this search?

        model = Model(
            pipeline=pipeline,
            dataset=dataset,
            target=target,
            symbol=symbol,
            features=feature_selection_method
        )
        model.parameters.append(mp)
        self.model_repo.create(model)

        # Save grid search results on storage
        if kwargs.get('save', True):
            self.storage.upload_json_obj(mp.parameters, 'grid-search-results', 'parameters-{}.json'.format(tag))
            self.storage.save_df(results_df, 'grid-search-results', mp.result_file)
        return mp


