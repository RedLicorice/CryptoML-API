from cryptoml_core.services.task_service import TaskService, Task
from fastapi import APIRouter, Body, HTTPException, Depends
from typing import Union, List
router = APIRouter()


@router.get('/')
def task_list(service: TaskService = Depends(TaskService)):
    return service.get_all()


@router.get('/result/{task_id}')
def task_result(task_id: str, service: TaskService = Depends(TaskService)):
    return service.get_result(task_id)

@router.post('/result')
def _task_result(task: Task = Body(...), service: TaskService = Depends(TaskService)):
    return service.get_result(task.task_id)

@router.get('/status/{task_id}')
def task_status(task_id: str, service: TaskService = Depends(TaskService)):
    return service.get_status(task_id)

@router.post('/status')
def task_check_many(request: Union[Task, List[Task]] = Body(...), service: TaskService = Depends(TaskService)):
    if isinstance(request, list):
        return [service.get_status(t.task_id).dict() for t in request]
    else:
        return service.get_status(request.task_id)

@router.post('/revoke')
def task_revoke(task: Task = Body(...), service: TaskService = Depends(TaskService)):
    return service.revoke(task)
