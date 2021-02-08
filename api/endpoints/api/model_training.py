from celery import current_app
import celery.states as states
from fastapi import APIRouter, Depends


router = APIRouter()

# Launch a Grid Search task using specified pipeline
@router.post('/grid_search')
def grid_search():

    task = current_app.send_task('gridsearch', args=['World'])
    if task.status != 'SUCCESS':
        return task.id
    return task.result

@router.get('/check/{task_id}')
def check(task_id: str):
    res = current_app.AsyncResult(task_id)
    return res.state if res.state == states.PENDING else str(res.result)