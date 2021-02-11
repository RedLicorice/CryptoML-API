from .lib.models import grid_search, train_model, test_model, predict_model
import pandas as pd
import importlib

# Launch a Distributed grid search on the cluster
def launch_grid_search(pipeline: str, features: pd.DataFrame, target: pd.Series, **kwargs):
    _pipeline = importlib.import_module('cryptoml.pipelines.{}'.format(pipeline))
    gscv = grid_search(
        est=_pipeline.estimator,
        parameters=kwargs.get('parameter_grid', _pipeline.PARAMETER_GRID),
        X_train=features.values,
        y_train=target.values,
        **kwargs
    )
    return gscv.best_params_
