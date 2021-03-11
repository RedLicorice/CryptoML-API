from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import PlainTextResponse
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.repositories.dataset_repository import DatasetRepository
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from celery import current_app

router = APIRouter()


@router.get('/')
def get_dataset(
        symbol: Optional[str] = None,
        service: DatasetService = Depends(DatasetService),
):
    if symbol:
        return [d for d in service.find_by_symbol(symbol)]
    else:
        return [d for d in service.all()]


@router.post('/merge/{name}/{symbol}')
def merge_dataset(
        name: str,
        symbol: str,
        sync: Optional[bool] = False,
        query: dict = Body(...),
        service: DatasetService = Depends(DatasetService),
        repo: DatasetRepository = Depends(DatasetRepository),
        tasks: TaskService = Depends(TaskService)
):
    if sync:
        datasets = repo.query(query)
        return service.merge_datasets(datasets, name, symbol)
    else:
        r = MergeRequest(
            query=query,
            name=name,
            symbol=symbol
        )
        return tasks.send(task_name='merge_datasets',
                          task_args=r.dict(),
                          name='merge_datasets-{}->{}-{}'.format(str(r.query), r.name, r.symbol),
                          batch=str(r.query))


class MergeRequest(BaseModel):
    query: dict
    name: str
    symbol: str


@router.post('/merge-many')
def merge_dataset_many(
        requests: List[MergeRequest] = Body(...),
        tasks: TaskService = Depends(TaskService)
):
    _tasks = [
        tasks.send(task_name='merge_datasets',
                   task_args=r.dict(),
                   name='merge_datasets-{}->{}-{}'.format(str(r.query), r.name, r.symbol),
                   batch=str(r.query))
        for r in requests
    ]
    return _tasks


@current_app.task(name='merge_datasets')
def task_build_dataset(req: dict):
    req = MergeRequest(**req)
    service: DatasetService = DatasetService()
    datasets = service.query(req.query)
    ds = service.merge_datasets(datasets, req.name, req.symbol)
    return ds.dict()


@router.get('/data/{dataset_id}', response_class=PlainTextResponse)
def get_dataset_csv(
        dataset_id: str,
        service: DatasetService = Depends(DatasetService),
):
    ds = service.get(dataset_id)
    df = service.get_features(name=ds.name, symbol=ds.symbol, begin=ds.index_min, end=ds.index_max, columns=ds.features)
    return df.to_csv(index_label='time')


@router.get('/data', response_class=PlainTextResponse)
def get_dataset(
        symbol: str,
        dataset: Optional[str] = None,
        target: Optional[str] = None,
        begin: Optional[str] = None,
        end: Optional[str] = None,
        service: DatasetService = Depends(DatasetService),
):
    if not dataset and not target:
        raise HTTPException(status_code=400,
                            detail="At least one of 'dataset' or 'target' parameters must be specified!")
    _name = dataset
    if not _name:
        _name = 'target'
    d = service.get_dataset(name=_name, symbol=symbol)
    # If begin/end not specified, use recorded.
    # If auto use valid.
    if not begin:
        begin = d.index_min
    elif begin == 'auto':
        begin = d.valid_index_min
    if not end:
        end = d.index_max
    elif end == 'auto':
        end = d.valid_index_max
    # Retrieve dataframes
    dfs = []
    if dataset:
        df = service.get_features(name=dataset, symbol=symbol, begin=begin, end=end)
        dfs.append(df)
    if target:
        dfs.append(service.get_target(name=target, symbol=symbol, begin=begin, end=end))
    # Concatenate dataframes and target
    res = pd.concat(dfs, axis='columns') if len(dfs) > 1 else dfs[0]
    # Return CSV
    return res.to_csv(index_label='time')
