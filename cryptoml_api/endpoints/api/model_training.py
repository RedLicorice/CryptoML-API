from celery import current_app
import celery.states as states
from fastapi import APIRouter, Depends, Form


router = APIRouter()

# Launch a Grid Search task using specified pipeline
@router.post('/grid_search')
def grid_search(
        pipeline: str = Form(...),
        symbol: str = Form(...),
        features: str = Form(...),
        target: str = Form(...)
    ):

    task = current_app.send_task('gridsearch', args=[pipeline, symbol, features, target])
    if task.status != 'SUCCESS':
        return task.id
    return task.result

@router.get('/check/{task_id}')
def check(task_id: str):
    res = current_app.AsyncResult(task_id)
    return res.state if res.state == states.PENDING else str(res.result)