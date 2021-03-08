from fastapi import APIRouter, Body, Depends, HTTPException
from cryptoml_core.exceptions import MessageException
from cryptoml_core.models.classification import Model, ModelTest
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.deps.celery import current_app, states

router = APIRouter()


@router.post('/test-sync/{model_id}')
def test_model_sync(model_id: str, test: ModelTest = Body(...), service: ModelService = Depends(ModelService)):
    model = service.get_model(model_id)
    return service.test_model(model, test)


@router.post('/test/{model_id}')
def test_model(model_id: str, test: ModelTest = Body(...), tasks: TaskService = Depends(TaskService), service: ModelService = Depends(ModelService)):
    try:
        # task = current_app.send_task('testmodel', args=[test.dict()])
        # if task.status != 'SUCCESS':
        #     return {'task': task.id}
        # return task.result
        model = service.get_model(model_id)
        return tasks.send(task_name='testmodel',
                          task_args={'model': model.dict(), 'test': test.dict()},
                          name='model_test-{}-{}-{}-{}'.format(model.symbol, model.pipeline, model.dataset, model.target)
                          )
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='testmodel')
def task_test_model(req: dict):
    model = Model(req['model'])
    test = ModelTest(req['test'])
    model_service = ModelService()
    mt = model_service.test_model(model, test)
    return mt.dict()
