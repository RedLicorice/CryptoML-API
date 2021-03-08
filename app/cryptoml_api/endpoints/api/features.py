from fastapi import APIRouter, Form, File, UploadFile, Depends, HTTPException, Body
from werkzeug.utils import secure_filename
from cryptoml_core.deps.celery import current_app
from cryptoml_core.services.storage_service import StorageService
from cryptoml_core.services.dataset_building_service import DatasetBuildingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.deps.config import config
from cryptoml_core.exceptions import MessageException
from pydantic import BaseModel
import logging
from typing import List, Optional

router = APIRouter()


class ImportRequest(BaseModel):
    bucket: str
    name: str
    dataset: str
    symbol: str


class BuildRequest(BaseModel):
    symbol: str
    builder: str
    args: dict


@router.post('/upload')
def upload(
        name: str = Form(...),
        symbol: str = Form(...),
        file: UploadFile = File(...),
        storage: StorageService = Depends(StorageService),
) -> ImportRequest:
    # CSV mime-type is "text/csv"
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail='You can only upload CSV files!')
    # Store uploaded dataset on S3
    filename = secure_filename(name)
    storage_name = "{}.{}.csv".format(symbol, filename)
    bucket = config['storage']['s3']['uploads_bucket'].get(str)
    try:
        storage.upload_file(file.file, bucket, storage_name)
        return ImportRequest(bucket=bucket, name=storage_name, dataset=name, symbol=symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to save file on storage')


@router.post('/import')
def _import(
        ticket: ImportRequest = Body(...),
        service: DatasetService = Depends(DatasetService)
):
    filename = secure_filename(ticket.name)
    if not ticket.dataset:
        raise HTTPException(status_code=400, detail='You must complete the dataset field!')
    # Import dataset to feature repository
    try:
        ds = service.import_from_storage(ticket.bucket, filename, ticket.dataset, ticket.symbol)
        return ds.dict()
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail='Failed to import data in repository')


@router.post('/import-many')
def _import_many(
        tickets: List[ImportRequest] = Body(...),
        service: DatasetService = Depends(DatasetService)
):
    results = []
    for ticket in tickets:
        filename = secure_filename(ticket.name)
        if not ticket.dataset:
            logging.error("Skipping {}: missing dataset!".format(ticket.symbol))
            continue
        # Import dataset to feature repository
        try:
            logging.info("Importing: {}".format(ticket.dict()))
            ds = service.import_from_storage(ticket.bucket, filename, ticket.dataset, ticket.symbol)
            results.append(ds.dict())
        except Exception as e:
            logging.exception(e)
            continue
    return results


@router.get('/build')
def get_builders(
        service: DatasetBuildingService = Depends(DatasetBuildingService)
):
    # Import dataset to feature repository
    try:
        return service.get_builders()
    except Exception as e:
        raise HTTPException(status_code=404, detail=['Data not found', e])


@router.post('/build')
def build_dataset(
        req: BuildRequest = Body(...),
        service: DatasetBuildingService = Depends(DatasetBuildingService),
        tasks: TaskService = Depends(TaskService)
):
    try:
        service.check_builder_args(req.builder, req.args)
        return tasks.send(task_name='build_dataset',
                          task_args=req.dict(),
                          name='build_dataset-{}-{}'.format(req.symbol, req.builder)
                          )
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post('/build-many')
def build_many_dataset(
        requests: List[BuildRequest] = Body(...),
        batch: Optional[str] = None,
        service: DatasetBuildingService = Depends(DatasetBuildingService),
        tasks: TaskService = Depends(TaskService)
):
    # Check all args are correct
    for req in requests:
        try:
            service.check_builder_args(req.builder, req.args)
        except MessageException as e:
            raise HTTPException(status_code=400, detail=e.message)
    # Launch tasks
    _tasks = [
        tasks.send(task_name='build_dataset',
                   task_args=r.dict(),
                   name='build_dataset-{}-{}'.format(req.symbol, req.builder),
                   batch=batch)
        for r in requests
    ]
    return _tasks


@current_app.task(name='build_dataset')
def task_build_dataset(req: dict):
    req = BuildRequest(**req)
    service: DatasetService = DatasetService()
    ds = service.build_dataset(req.symbol, req.builder, req.args)
    return ds.dict()
