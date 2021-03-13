from fastapi import APIRouter, Body, Depends, HTTPException
from cryptoml_core.exceptions import MessageException
from cryptoml_core.models.classification import Model, ModelTest
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.deps.celery import current_app, states
from typing import Optional

router = APIRouter()


@router.get('/')
def model_index(service: ModelService = Depends(ModelService)):
    return service.all()


@router.get('/create')
def model_index(query: Optional[dict] = Body(...), service: ModelService = Depends(ModelService)):
    return service.create_classification_models(query)


@router.post('/clear')
def model_index(
        features: Optional[bool] = True,
        parameters: Optional[bool] = True,
        tests: Optional[bool] = True,
        query: Optional[dict] = Body(...),
        service: ModelService = Depends(ModelService)
):
    result = {}
    if features:
        result['features'] = service.clear_features(query)
    if parameters:
        result['parameters'] = service.clear_parameters(query)
    if tests:
        result['tests'] = service.clear_tests(query)
    return result


@router.post('/test/{model_id}')
def test_model(
        model_id: str,
        sync: Optional[bool] = False,
        test: ModelTest = Body(...),
        tasks: TaskService = Depends(TaskService),
        service: ModelService = Depends(ModelService)
):
    try:
        model = service.get_model(model_id)
        if sync:
            return service.test_model(model, test)
        return tasks.send(task_name='testmodel',
                          task_args={'model': model.dict(), 'test': test.dict()},
                          name='model_test-{}-{}-{}-{}'.format(model.symbol, model.pipeline, model.dataset,
                                                               model.target)
                          )
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='testmodel')
def task_test_model(req: dict):
    model = Model(**req['model'])
    test = ModelTest(**req['test'])
    model_service = ModelService()
    mt = model_service.test_model(model, test)
    return mt.dict()
