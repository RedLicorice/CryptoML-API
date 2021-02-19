from celery import current_app
import celery.states as states
from fastapi import APIRouter, Body, HTTPException
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.exceptions import MessageException
from cryptoml_core.models.classification import SplitClassification


router = APIRouter()

@router.post('/gridsearch-sync')
def grid_search(
        req: SplitClassification = Body(...)
    ):
    try:
        model_service = ModelService()
        gscv_df, params, report = model_service.grid_search(req)
        return {'parameters':params, 'report':report}
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)

# Launch a Grid Search task using specified pipeline
@router.post('/gridsearch')
def grid_search(
        req: SplitClassification = Body(...)
    ):
    try:
        task = current_app.send_task('gridsearch', args=[req.dict()])
        if task.status != 'SUCCESS':
            return {'task': task.id}
        return task.result
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='gridsearch')
def grid_search(req: dict):
    req = SplitClassification(**req)
    # We want to optimize precision score for both BUY and SELL
    model_service = ModelService()
    gscv_df, params = model_service.grid_search(req)

    return params

@router.get('/gridsearch/{task_id}')
def check(task_id: str):
    res = current_app.AsyncResult(task_id)
    return res.state if res.state == states.PENDING else res.result