import typer
from cryptoml_core.services.grid_search import GridSearchService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging


def main(dataset: str, target: str, pipeline: str, features: Optional[str] = None, halving: Optional[bool] = False, save: Optional[bool] = True):
    service = GridSearchService()
    models = ModelService()
    query = {"dataset": dataset, "target": target, "pipeline": pipeline}
    if pipeline == 'all':
        del query['pipeline']
    if target == 'all':
        del query['target']
    if save:
        models.clear_parameters(query)
    else:
        logging.info("Results will not  be saved [--no-save]")
    search_models = models.query_models(query)
    logging.info("[i] {} models to train".format(len(search_models)))
    for i, m in enumerate(search_models):
        logging.info("==[{}/{}]== MODEL: {} {} {} {} =====".format(i+1, len(search_models), m.symbol, m.dataset, m.target, m.pipeline))
        mp = service.create_parameters_search(m, split=0.7, features=features)
        logging.info("[{}] Start grid search".format(get_timestamp()))
        mp = service.grid_search(m, mp, sync=True, verbose=1, n_jobs=8, halving=halving, save=save)
        logging.info("[{}] End grid search".format(get_timestamp()))


if __name__ == '__main__':
    setup_file_logger('grid_search.log')
    typer.run(main)
