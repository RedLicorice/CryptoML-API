from celery import current_app
import celery.states as states
from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel
from ...services.feature_service import FeatureService
from ...services.model_service import ModelService
from ...services.storage_service import StorageService
from ...exceptions import MessageException
from cryptoml.util.weighted_precision_score import get_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit


router = APIRouter()

class GridSearchRequest(BaseModel):
    pipeline: str
    symbol: str
    dataset: str
    target: str

# Launch a Grid Search task using specified pipeline
@router.post('/gridsearch')
def grid_search(
        req: GridSearchRequest = Body(...),
        models: ModelService = Depends(ModelService),
        features: FeatureService = Depends(FeatureService)
    ):
    # Make sure that dataset and target exist for this symbol
    try:
        features.classification_exists(req.symbol, req.dataset, req.target)
    except MessageException as e:
        raise HTTPException(status_code=404, detail=e.message)
    # Check the pipeline exists and is valid
    try:
        models.get_pipeline(req.pipeline)
    except MessageException as e:
        raise HTTPException(status_code=404, detail=e.message)
    # Launch the grid search task
    try:
        task = current_app.send_task('gridsearch', args=[req.dict()])
        if task.status != 'SUCCESS':
            return {'task': task.id}
        return task.result
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)

@current_app.task(name='gridsearch')
def grid_search(req: dict):
    req = GridSearchRequest(**req)
    # Retrieve feature set from repository
    features = FeatureService()
    X_train, X_test, y_train, y_test = features.get_classification(req.symbol, req.dataset, req.target, split=0.7)
    # We want to optimize precision score for both BUY and SELL
    model_service = ModelService()
    splitter = BlockingTimeSeriesSplit(n_splits=5)
    scorer = get_scorer(weights={0: 1.0, 1:0.7, 2:1.0})
    gscv_df, params = model_service.grid_search(
        pipeline=req.pipeline,
        features=X_train,
        target=y_train,
        cv=splitter,
        scoring=scorer
    )
    # Store grid search results on storage
    tag = '{}-{}-{}-{}'.format(req.pipeline, req.symbol, req.dataset, req.target)
    storage: StorageService = StorageService()
    storage.upload_json_obj(params, 'grid-search-results', 'parameters-{}.json'.format(tag))
    storage.save_df(gscv_df, 'grid-search-results', 'cv_results-{}.csv'.format(tag))
    # Test model parameters on the test set
    for _w in [30, 90, 150]:
        clf_report_df = model_service.test_model(req.pipeline, params, X_train, X_test, y_train, y_test, W=_w)
        if clf_report_df:
            storage.save_df(clf_report_df, 'grid-search-results', 'test_report-W{}-{}.csv'.format(_w, tag))
    return params

@router.get('/gridsearch/{task_id}')
def check(task_id: str):
    res = current_app.AsyncResult(task_id)
    return res.state if res.state == states.PENDING else res.result