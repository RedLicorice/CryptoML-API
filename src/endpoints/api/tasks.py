from celery import current_app as app
import celery.states as states
from fastapi import APIRouter, Depends
from src.services.FeaturesService import FeaturesService
from src.services.StorageService import StorageService


router = APIRouter()

@router.get('/grid_search')
def task(
        fs: FeaturesService = Depends(FeaturesService),
        ss: StorageService = Depends(StorageService)
    ):
    a= 10
    b = 20
    celery = app
    print("HELLO: {} {} {} {}".format(a,b,celery,fs,ss))
    task = celery.send_task('gridsearch', args=['World'])
    print('Hello task submitted')
    if task.status != 'SUCCESS':
        return task.id
    return task.result

@router.get('/check/{task_id}')
def check(task_id: str):
    res = app.AsyncResult(task_id)
    return res.state if res.state == states.PENDING else str(res.result)