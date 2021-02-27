from cryptoml_core.deps.celery import current_app, states
from fastapi import APIRouter, Body, HTTPException, Depends
from cryptoml_core.services.tuning_service import TuningService, GridSearch, ModelTest, ModelTestBlueprint
from cryptoml_core.services.task_service import TaskService, Task
from cryptoml_core.exceptions import MessageException
from pydantic import BaseModel
from typing import Optional, List, Union

router = APIRouter()



class GridSearchRequest(BaseModel):
    task: GridSearch
    tests: Optional[ModelTestBlueprint] = None # List of windows on which tests should be spawned

class GridSearchResponse(BaseModel):
    result: GridSearch
    tests: Optional[Union[List[ModelTest], List[Task]]]  # List of tasks started for testing this model

@router.post('/gridsearch-sync')
def grid_search(
        req: GridSearchRequest = Body(...),
        service: TuningService = Depends(TuningService)
    ):
    try:
        gs = service.grid_search(req.task, sync=True)
        res = GridSearchResponse(result=gs)
        if req.tests:
            res.tests = [t for t in service.tests_from_blueprint(gs, req.tests)] # If sync, return modeltest list
        return res.dict()
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


# Launch a Grid Search task using specified pipeline
@router.post('/gridsearch')
def grid_search(
        req: GridSearchRequest = Body(...),
        tasks: TaskService = Depends(TaskService)
    ):
    try:
        # task = current_app.send_task('gridsearch', args=[req.dict()])
        # if task.status != 'SUCCESS':
        #     return {'task': task.id}
        # return task.result
        return tasks.send('gridsearch', args=req.dict())
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='gridsearch')
def grid_search(req: dict):
    req = GridSearchRequest(**req)
    service = TuningService()
    gs = service.grid_search(req.task)
    res = GridSearchResponse(result=gs)
    if req.tests:
        tasks = TaskService()
        tests = [t.dict() for t in service.tests_from_blueprint(gs, req.tests)]  # Returns list of ModelTest, to dict
        res.tests = [t for t in tasks.send_many('testmodel', args_list=tests)]  # Returns list of Tasks
        # res.tests = {str(t.window): current_app.send_task('testmodel', args=[t.dict()]).id for t in }
    return res.dict()

