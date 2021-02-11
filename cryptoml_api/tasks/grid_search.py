import time
from celery import current_app as app
from ..services import StorageService
from ..repositories import FeatureRepository
from ..dask import dask_client
from sklearn.metrics import make_scorer, precision_score, classification_report
from cryptoml.api import launch_grid_search
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit

@app.task(name='gridsearch')
def grid_search(pipeline: str, symbol:str, features: str, target: str):
    # Retrieve feature set from repository
    repo: FeatureRepository = FeatureRepository()
    _features = repo.get_dataframe(symbol, features)
    _target = repo.get_series(symbol, target)
    # We want to optimize precision score for both BUY and SELL
    splitter = BlockingTimeSeriesSplit(n_splits=5)
    scorer = make_scorer(precision_score, labels=[0, 1, 2], pos_label=[0, 2], average='weighted', zero_division=0)
    gscv = launch_grid_search(pipeline, _features, _target, scoring=scorer, cv=splitter)
    # Store grid search results on storage
    params = gscv.best_params__
    storage: StorageService = StorageService()
    storage.upload_json_obj(params, 'grid-search-results', 'parameters-{}-{}-{}-{}.json'.format(
        pipeline, symbol, features, target
    ))