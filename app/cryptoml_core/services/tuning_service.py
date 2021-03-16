import pandas as pd
# APP Dependencies
from .storage_service import StorageService
from .dataset_service import DatasetService
from ..exceptions import MessageException
# CryptoML Lib Dependencies
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer, get_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit
from cryptoml.util.import_proxy import GridSearchCV
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
from cryptoml_core.deps.dask import get_client
from cryptoml_core.models.classification import Model, ModelParameters, ModelFeatures
from cryptoml_core.repositories.classification_repositories import ModelRepository
# from cryptoml_core.models.tuning import GridSearch, ModelTestBlueprint, ModelTest
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.util.dict_hash import dict_hash
from cryptoml.util.feature_importances import label_feature_importances, label_support
from sklearn.utils import parallel_backend
from sklearn.feature_selection import SelectFromModel, RFECV, SelectPercentile, f_classif
from cryptoml_core.exceptions import NotFoundException
from dask.distributed import CancelledError
from uuid import uuid4
import logging
from skrebate import ReliefF, MultiSURF
import math
import numpy as np


def select_rfecv(X, y, sync=False):
    # Load pipeline
    pipeline_module = get_pipeline("selection_logistic", unlisted=True)

    # Perform search
    rfecv = RFECV(
        estimator=pipeline_module.estimator,
        cv=BlockingTimeSeriesSplit(n_splits=5),
        scoring=get_precision_scorer(),
        importance_getter='named_steps.c.coef_'
    )
    if sync:
        rfecv.fit(X, y)
    else:
        dask = get_client()  # Connect to Dask scheduler
        with parallel_backend('dask'):
            rfecv.fit(X, y)
    return rfecv


def select_from_model(X, y, sync=False):
    # Load pipeline
    pipeline_module = get_pipeline("selection_xgboost", unlisted=True)

    # Perform search
    sfm = SelectFromModel(
        pipeline_module.estimator,
        threshold='mean',
        importance_getter='named_steps.c.feature_importances_'
    )
    if sync:
        sfm.fit(X, y)
    else:
        dask = get_client()  # Connect to Dask scheduler
        with parallel_backend('dask'):
            sfm.fit(X, y)
    return sfm


def select_percentile(X, y, percentile=10):
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    selector.fit(X, y)
    return selector


# NOTE: ReliefF (and MultiSURF) expect n_neighbors (k) is <= to the number of instances
# that have the least frequent class label (binary and multiclass endpoint data)
def select_relieff(X, y, percentile=10):
    unique, counts = np.unique(y, return_counts=True)
    num = math.ceil(X.shape[0] * percentile / 100)
    k = np.min(counts)
    if k > 100:
        k = 100
    selector = ReliefF(
        n_features_to_select=num,
        n_neighbors=k,
        discrete_threshold=3,
        n_jobs=-1
    )
    selector.fit(X, y)
    return selector


def select_multisurf(X, y, percentile=10):
    num = math.ceil(X.shape[0] * percentile / 100)
    selector = MultiSURF(
        n_features_to_select=num,
        discrete_threshold=3,
        n_jobs=-1
    )
    selector.fit(X, y)
    return selector

class TuningService:
    def __init__(self):
        self.storage = StorageService()
        self.model_repo = ModelRepository()
        self.dataset_service = DatasetService()

    def create_parameters_search(self, model: Model, split: float, task_key: str = None, **kwargs) -> ModelParameters:
        ds = self.dataset_service.get_dataset(model.dataset, model.symbol)
        splits = self.dataset_service.get_train_test_split_indices(ds, split)

        features = kwargs.get('features')
        if isinstance(features, str) and features == 'latest':
            if model.features:
                features = model.features[-1].features
            else:
                features = None

        result = ModelParameters(
            cv_interval=splits['train'],
            cv_splits=5,
            task_key=task_key or str(uuid4()),
            features=features or None
        )
        return result

    def grid_search(self, model: Model, mp: ModelParameters, **kwargs) -> ModelParameters:
        if not model.id:  # Make sure the task exists
            model = self.model_repo.create(model)
        if self.model_repo.exist_parameters(model.id, mp.task_key):
            logging.info("Model {} Grid search {} already executed!".format(model.id, mp.task_key))
            return mp
        # Load dataset
        ds = DatasetService()
        X = ds.get_features(model.dataset, model.symbol, mp.cv_interval.begin, mp.cv_interval.end, columns=mp.features)
        y = ds.get_target(model.target, model.symbol, mp.cv_interval.begin, mp.cv_interval.end)

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)

        # Perform search
        gscv = GridSearchCV(
            estimator=pipeline_module.estimator,
            param_grid=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
            cv=BlockingTimeSeriesSplit(n_splits=mp.cv_splits),
            scoring=get_precision_scorer(),
            verbose=kwargs.get("verbose", 0),
            n_jobs=kwargs.get("n_jobs", None)
        )

        mp.start_at = get_timestamp()  # Log starting timestamp
        if kwargs.get('sync', False):
            gscv.fit(X, y)
        else:
            # Only run parallel backend if not sync
            dask = get_client()  # Connect to Dask scheduler
            with parallel_backend('dask'):
                gscv.fit(X, y)
        mp.end_at = get_timestamp()  # Log ending timestamp

        # Collect results
        results_df = pd.DataFrame(gscv.cv_results_)

        # Update search request with results
        mp.parameter_search_method = 'gridsearch'
        mp.parameters = gscv.best_params_
        tag = "{}-{}-{}-{}-{}"\
            .format(model.symbol, model.dataset, model.target, model.pipeline, dict_hash(mp.parameters))
        mp.result_file = 'cv_results-{}.csv'.format(tag)

        # Store grid search results on storage
        self.storage.upload_json_obj(mp.parameters, 'grid-search-results', 'parameters-{}.json'.format(tag))
        self.storage.save_df(results_df, 'grid-search-results', mp.result_file)

        # Update model with the new results
        self.model_repo.append_parameters(model.id, mp)
        return mp

    def create_features_search(self, *, model: Model, split: float, method: str, task_key: str = None) -> ModelFeatures:
        ds = self.dataset_service.get_dataset(model.dataset, model.symbol)
        splits = self.dataset_service.get_train_test_split_indices(ds, split)
        result = ModelFeatures(
            search_interval=splits['train'],
            feature_selection_method=method,
            task_key=task_key or str(uuid4())
        )
        return result

    def feature_selection(self, model: Model, mf: ModelFeatures, **kwargs) -> ModelFeatures:
        INDEPENDENT_SEARCH_METHODS = ['importances', 'rfecv', 'fscore', 'relieff', 'multisurf']
        if not model.id:  # Make sure the model exists
            model = self.model_repo.create(model)
        if self.model_repo.exist_features(model.id, mf.task_key):
            logging.info("Model {} Feature selection {} already executed!".format(model.id, mf.task_key))
            return mf
        # Load dataset
        ds = DatasetService()
        X = ds.get_features(model.dataset, model.symbol, mf.search_interval.begin, mf.search_interval.end, columns=mf.features)
        y = ds.get_target(model.target, model.symbol, mf.search_interval.begin, mf.search_interval.end)

        # Perform search
        mf.start_at = get_timestamp()  # Log starting timestamp
        if not mf.feature_selection_method or mf.feature_selection_method == 'importances':
            selector = select_from_model(X, y, sync=kwargs.get('sync', False))
            mf.feature_importances = label_feature_importances(selector.estimator_, X.columns)
        elif mf.feature_selection_method == 'rfecv':
            selector = select_rfecv(X, y, sync=kwargs.get('sync', False))
        elif mf.feature_selection_method == 'fscore':
            selector = select_percentile(X, y, percentile=10)
        elif mf.feature_selection_method == 'relieff':
            selector = select_relieff(X, y, percentile=10)
        elif mf.feature_selection_method == 'multisurf':
            selector = select_multisurf(X, y, percentile=10)
        else:
            raise NotFoundException("Cannot find feature selection method by {}".format(mf.feature_selection_method))
        mf.end_at = get_timestamp()  # Log ending timestamp

        # Update search request with results
        mf.features = label_support(selector.get_support(), X.columns)

        # Update model with the new results
        if mf.feature_selection_method in INDEPENDENT_SEARCH_METHODS:
            self.model_repo.append_features_query(
                {"dataset": model.dataset, "symbol": model.symbol, "target": model.target},
                mf
            )
        else:
            self.model_repo.append_features(model.id, mf)
        return mf






