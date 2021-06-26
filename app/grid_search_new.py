import typer
from cryptoml_core.services.grid_search import GridSearchService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging
import json
from joblib import cpu_count


def main(symbol: str, dataset: str, target: str, pipeline: str, feature_selection_method: Optional[str] = 'importances_shap', split: Optional[float] = 0.7, replace: Optional[bool] = True, save: Optional[bool] = True):
    service = GridSearchService()
    n_jobs = int(cpu_count() / 2)
    multithread_pipeline = ['mlp', 'xgboost']
    if any(ext in pipeline for ext in multithread_pipeline):
        n_jobs = int(n_jobs / 2 + 1)
    logging.info("[{}] {}({}.{}) -> {} Start grid search (JOBS: {})".format(get_timestamp(), pipeline, dataset, symbol, target, n_jobs))
    mp = service.grid_search_new(
        pipeline=pipeline,
        dataset=dataset,
        target=target,
        symbol=symbol,
        split=split,
        feature_selection_method=feature_selection_method,
        verbose=1,
        n_jobs=n_jobs,
        replace=replace,
        save=save
    )
    logging.info("[{}] End grid search\n".format(get_timestamp()))


if __name__ == '__main__':
    setup_file_logger('grid_search_new.log')
    typer.run(main)
