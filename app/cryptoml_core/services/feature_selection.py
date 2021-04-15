from cryptoml.pipelines import get_pipeline
from cryptoml.util.feature_importances import label_feature_importances, label_support
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.deps.dask import get_client
from cryptoml_core.models.classification import Model, ModelParameters, ModelFeatures
from cryptoml_core.repositories.classification_repositories import ModelRepository
from cryptoml_core.exceptions import MessageException, NotFoundException
from cryptoml_core.util.timestamp import get_timestamp
from sklearn.model_selection import GridSearchCV
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit
from sklearn.utils import parallel_backend
from skrebate import ReliefF, MultiSURF
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
from cryptoml.util.flattened_classification_report import classification_report_imbalanced
import numpy as np
import math
import logging
from uuid import uuid4


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


def select_from_model_cv(X, y, sync=False):
    # Load pipeline
    pipeline_module = get_pipeline("selection_xgboost", unlisted=True)
    gscv = GridSearchCV(
        estimator=pipeline_module.estimator,
        param_grid=pipeline_module.PARAMETER_GRID,
        scoring='precision',
        cv=3,
        n_jobs=8
    )

    # Fit grid search
    gscv.fit(X, y)

    # Perform search
    sfm = SelectFromModel(
        gscv,
        # prefit=True,
        threshold='mean',
        importance_getter='best_estimator_.named_steps.c.feature_importances_'
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


class FeatureSelectionService:
    def __init__(self):
        self.model_repo = ModelRepository()
        self.dataset_service = DatasetService()

    def create_features_search(self, *, symbol: str, dataset: str, target: str, split: float, method: str, task_key: str = None) -> ModelFeatures:
        ds = self.dataset_service.get_dataset(dataset, symbol)
        splits = self.dataset_service.get_train_test_split_indices(ds, split)
        result = ModelFeatures(
            dataset=dataset,
            target=target,
            symbol=symbol,
            search_interval=splits['train'],
            feature_selection_method=method,
            task_key=task_key or str(uuid4())
        )
        return result

    def feature_selection(self, mf: ModelFeatures, **kwargs) -> ModelFeatures:

        # Load dataset
        X = self.dataset_service.get_features(mf.dataset, mf.symbol, mf.search_interval.begin, mf.search_interval.end,
                            columns=mf.features)
        y = self.dataset_service.get_target(mf.target, mf.symbol, mf.search_interval.begin, mf.search_interval.end)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(mf.symbol, mf.dataset, mf.target, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))

        # Perform search
        mf.start_at = get_timestamp()  # Log starting timestamp
        if not mf.feature_selection_method or mf.feature_selection_method == 'importances':
            selector = select_from_model(X, y, sync=kwargs.get('sync', False))
            mf.feature_importances = label_feature_importances(selector.estimator_, X.columns)
        elif mf.feature_selection_method == 'importances_cv':
            selector = select_from_model_cv(X, y, sync=kwargs.get('sync', False))
            mf.feature_importances = label_feature_importances(selector.estimator_.best_estimator_, X.columns)
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
        if kwargs.get('save', True):
            self.model_repo.append_features_query(
                {"dataset": mf.dataset, "symbol": mf.symbol, "target": mf.target},
                mf
            )
        return mf