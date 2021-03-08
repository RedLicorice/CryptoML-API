from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from cryptoml_core.services.dataset_service import DatasetService
from typing import Optional
import pandas as pd

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
        raise HTTPException(status_code=400, detail="At least one of 'dataset' or 'target' parameters must be specified!")
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

