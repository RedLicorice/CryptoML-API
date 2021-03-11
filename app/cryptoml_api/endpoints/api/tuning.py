from cryptoml_core.deps.celery import current_app, states
from fastapi import APIRouter, Body, HTTPException, Depends
from cryptoml_core.models.classification import Model, ModelParameters, ModelFeatures
from cryptoml_core.services.tuning_service import TuningService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.task_service import TaskService
from cryptoml_core.exceptions import MessageException
from typing import Optional, List

router = APIRouter()


#####
#  Grid Search Hyperparameter tuning
####

# Launch a Grid Search task using specified pipeline
@router.post('/gridsearch/{model_id}')
def grid_search(
        model_id: str,
        split: Optional[float] = 0.7,
        batch: Optional[str] = None,
        sync: Optional[bool] = False,
        model_service: ModelService = Depends(ModelService),
        tuning_service: TuningService = Depends(TuningService),
        tasks: TaskService = Depends(TaskService)
):
    try:
        model = model_service.get_model(model_id)
        parameters = tuning_service.create_parameters_search(model, split)
        if sync:
            return tuning_service.grid_search(model, parameters, sync=True)
        return tasks.send(task_name='gridsearch',
                          task_args={'model': model.dict(), 'search_parameters': parameters.dict()},
                          name='grid_search-{}-{}-{}-{}'.format(model.symbol, model.pipeline, model.dataset,
                                                                model.target),
                          batch=batch)
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post('/gridsearch-batch/')
def grid_search_batch(
        batch: Optional[str] = None,
        split: Optional[float] = 0.7,
        query: dict = Body(...),
        model_service: ModelService = Depends(ModelService),
        tuning_service: TuningService = Depends(TuningService),
        tasks: TaskService = Depends(TaskService)
):
    try:
        models = model_service.query_models(query)
        tests = [(model, tuning_service.create_parameters_search(model, split)) for model in models]
        return [tasks.send(task_name='gridsearch',
                           task_args={'model': model.dict(), 'search_parameters': search_parameters.dict()},
                           name='grid_search-{}-{}-{}-{}'.format(model.symbol, model.pipeline,
                                                                 model.dataset, model.target),
                           batch=batch,
                           countdown=30) for model, search_parameters in tests]
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='gridsearch')
def task_grid_search(req: dict):
    model = Model(**req['model'])
    search_parameters = ModelParameters(**req['search_parameters'])
    service = TuningService()
    res = service.grid_search(model, search_parameters)
    return res.dict()


#####
#  Feature selection
####
@router.post('/featureselection/{model_id}')
def feature_selection(
        model_id: str,
        method: str,
        split: Optional[float] = 0.7,
        batch: Optional[str] = None,
        sync: Optional[bool] = False,
        model_service: ModelService = Depends(ModelService),
        tuning_service: TuningService = Depends(TuningService),
        tasks: TaskService = Depends(TaskService)
):
    try:
        model = model_service.get_model(model_id)
        mf = tuning_service.create_features_search(model, split, method)
        if sync:
            return tuning_service.feature_selection(model, mf, sync=True)
        return tasks.send(task_name='featureselection',
                          task_args={'model': model.dict(), 'search_parameters': mf.dict()},
                          name='feature_selection-{}-{}-{}-{}'.format(model.symbol, model.pipeline, model.dataset,
                                                                      model.target),
                          batch=batch)
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post('/featureselection-batch/')
def feature_selection_batch(
        method: str,
        batch: Optional[str] = None,
        split: Optional[float] = 0.7,
        query: dict = Body(...),
        model_service: ModelService = Depends(ModelService),
        tuning_service: TuningService = Depends(TuningService),
        tasks: TaskService = Depends(TaskService)
):
    try:
        models = model_service.query_models(query)
        # This will only keep 1 copy for each (symbol, dataset, target) tuple
        d_models = {'{}-{}-{}'.format(m.symbol, m.dataset, m.target): m for m in models}
        models = [v for k, v in d_models.items()]

        def get_name_from_model(_model):
            return 'feature_selection-{}-{}-{}-{}'.format(
                _model.symbol, _model.pipeline, _model.dataset, _model.target)

        tests = [(model, tuning_service.create_features_search(model, split, method)) for model in models]
        return [tasks.send(task_name='featureselection',
                           task_args={'model': model.dict(), 'search_parameters': search_parameters.dict()},
                           name=get_name_from_model(model),
                           batch=batch,
                           countdown=30) for i, (model, search_parameters) in enumerate(tests)]
    except MessageException as e:
        raise HTTPException(status_code=400, detail=e.message)


@current_app.task(name='featureselection')
def task_feature_selection(req: dict):
    model = Model(**req['model'])
    search_parameters = ModelFeatures(**req['search_parameters'])
    service = TuningService()
    res = service.feature_selection(model, search_parameters)
    return res.dict()
