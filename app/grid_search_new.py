import typer
from cryptoml_core.services.grid_search import GridSearchService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging
import json


def main(symbol: str, dataset: str, target: str, pipeline: str, feature_selection_method: Optional[str] = 'importances_shap', split: Optional[float] = 0.7, save: Optional[bool] = True):
    service = GridSearchService()

    logging.info("[{}] Start grid search".format(get_timestamp()))
    mp = service.grid_search_new(
        pipeline=pipeline,
        dataset=dataset,
        target=target,
        symbol=symbol,
        split=split,
        feature_selection_method=feature_selection_method,
        verbose=1,
        n_jobs=8,
        save=save
    )
    logging.info("[{}] End grid search".format(get_timestamp()))


if __name__ == '__main__':
    setup_file_logger('grid_search_new.log')
    typer.run(main)
