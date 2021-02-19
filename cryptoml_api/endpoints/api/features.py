from fastapi import APIRouter, Form, File, UploadFile, Depends, HTTPException, Body
from fastapi.responses import PlainTextResponse
from werkzeug.utils import secure_filename
from typing import Optional
from celery import current_app, states
from cryptoml_core.services.storage_service import StorageService
from cryptoml_core.services.feature_service import FeatureService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.deps.config import config
from cryptoml_core.exceptions import MessageException
from pydantic import BaseModel
import logging


router = APIRouter()

class ImportRequest(BaseModel):
    bucket: str
    name: str
    dataset: str
    symbol: str

class DisplayRequest(BaseModel):
    dataset: Optional[str] = None
    target: Optional[str] = None
    symbol: Optional[str]
    begin: Optional[str] = None
    end: Optional[str] = None

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
    storage_name = "{}.{}.csv".format(symbol,filename)
    bucket = config['storage']['s3']['uploads_bucket'].get(str)
    try:
        storage.upload_file(file.file, bucket, storage_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to save file on storage')
        # return {'message': 'Failed to save data on s3 storage', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
    # Return dataset spec
    return ImportRequest(bucket=bucket, name=storage_name, dataset="", symbol=symbol)

@router.post('/import')
def _import(
        ticket: ImportRequest = Body(...),
        features: FeatureService = Depends(FeatureService)
    ) -> DisplayRequest:
    filename = secure_filename(ticket.name)
    if not ticket.dataset:
        raise HTTPException(status_code=400, detail='You must complete the dataset field!')
    # Import dataset to feature repository
    try:
        features.import_from_storage(ticket.bucket, filename, ticket.dataset, ticket.symbol)
    except Exception as e:
        #return {'message': 'Failed to import CSV in feature repository', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
        raise HTTPException(status_code=500, detail='Failed to import data in repository')
    return DisplayRequest(dataset=ticket.dataset, symbol=ticket.symbol)

@router.get('/view', response_class=PlainTextResponse)
def get_dataset(
        req: DisplayRequest = Body(...),
        service: DatasetService = Depends(DatasetService)
    ):
    # Import dataset to feature repository
    if not req.dataset and not req.target:
        raise HTTPException(status_code=400, detail='Must specify at least one of dataset or target')
    try:
        return service.get_dataset(**req.dict()).to_csv(index_label='time')
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=404, detail='Data not found')

@router.get('/build')
def get_builders(
        service: FeatureService = Depends(FeatureService)
    ):
    # Import dataset to feature repository
    try:
        return service.get_builders()
    except Exception as e:
        raise HTTPException(status_code=404, detail=['Data not found', e])

@router.post('/build')
def build_dataset(
        req: BuildRequest = Body(...),
        service: FeatureService = Depends(FeatureService)
    ):
    try:
        service.check_builder_args(req.builder, req.args)
        task = current_app.send_task('build_dataset', args=[req.dict()])
        if task.status != 'SUCCESS':
            return {'task':task.id}
        return task.result
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)

@current_app.task(name='build_dataset')
def task_build_dataset(req: dict):
    req = BuildRequest(**req)
    try:
        service: FeatureService = FeatureService()
        features = service.build_dataset(req.symbol, req.builder, req.args, store=True)
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)
    return DisplayRequest(dataset=req.builder, symbol=req.symbol).dict()

@router.get('/build/{task_id}')
def get_build_status(
        task_id: str,
    ):
    try:
        res = current_app.AsyncResult(task_id)
        return res.state if res.state == states.PENDING else str(res.result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=['Data not found', e])