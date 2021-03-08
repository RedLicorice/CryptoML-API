from cryptoml_core.deps.celery import current_app, states
from fastapi import APIRouter, Body, HTTPException, Depends
from cryptoml_core.models.classification import Model, ModelParameters
from cryptoml_core.services.tuning_service import TuningService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.exceptions import MessageException
from typing import Optional, List

router = APIRouter()


# class GridSearchRequest(BaseModel):
#     task: GridSearch
#     tests: Optional[ModelTestBlueprint] = None # List of windows on which tests should be spawned
#
# class GridSearchResponse(BaseModel):
#     result: GridSearch
#     tests: Optional[Union[List[ModelTest], List[Task]]]  # List of tasks started for testing this model

@router.post('/gridsearch-sync/{model_id}')
def grid_search(
        model_id: str,
        search_parameters: ModelParameters = Body(...),
        service: TuningService = Depends(TuningService),
        model_service: ModelService = Depends(ModelService)
    ):
    try:
        model = model_service.get_model(model_id)
        return service.grid_search(model, search_parameters)
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


# Launch a Grid Search task using specified pipeline
@router.post('/gridsearch/{model_id}')
def grid_search(
        model_id: str,
        search_parameters: ModelParameters = Body(...),
        batch: Optional[str] = None,
        model_service: ModelService = Depends(ModelService),
        tasks: TaskService = Depends(TaskService)
    ):
    try:
        model = model_service.get_model(model_id)
        return tasks.send(task_name='gridsearch',
                          task_args={'model': model.dict(), 'parameters': search_parameters.dict()},
                          name='grid_search-{}-{}-{}-{}'.format(model.symbol, model.pipeline, model.dataset, model.target),
                          batch=batch)
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post('/gridsearch-batch/{model_id}')
def grid_search_many(
        model_id: str,
        req: List[ModelParameters] = Body(...),
        batch: Optional[str] = None,
        model_service: ModelService = Depends(ModelService),
        tasks: TaskService = Depends(TaskService)
        ):
    try:
        model = model_service.get_model(model_id)
        return [tasks.send(task_name='gridsearch',
                      task_args={'model': model.dict(), 'search_parameters': search_parameters.dict()},
                      name='grid_search-{}-{}-{}-{}'.format(model.symbol, model.pipeline,
                                                            model.dataset, model.target),
                      batch=batch,
                      countdown=30) for search_parameters in req]
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='gridsearch')
def grid_search(req: dict):
    model = Model(req['model'])
    search_parameters = ModelParameters(req['search_parameters'])
    service = TuningService()
    res = service.grid_search(model, search_parameters)
    return res

