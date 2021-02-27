from fastapi import APIRouter, Body, Depends, HTTPException
from cryptoml_core.exceptions import MessageException
from cryptoml_core.models.tuning import ModelTest
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.deps.celery import current_app, states

router = APIRouter()


@router.post('/test-sync')
def test_model_sync(test: ModelTest = Body(...), service: ModelService = Depends(ModelService)):
    return service.test_model(test)


@router.post('/test')
def test_model(test: ModelTest = Body(...), tasks: TaskService = Depends(TaskService)):
    try:
        # task = current_app.send_task('testmodel', args=[test.dict()])
        # if task.status != 'SUCCESS':
        #     return {'task': task.id}
        # return task.result
        return tasks.send('testmodel', args=test.dict())
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='testmodel')
def task_test_model(req: dict):
    req = ModelTest(**req)
    model_service = ModelService()
    mt = model_service.test_model(req)
    return mt.dict()
