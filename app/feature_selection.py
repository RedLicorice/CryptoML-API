import typer
from cryptoml_core.services.feature_selection import FeatureSelectionService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.logging import setup_file_logger
import logging


def main(dataset: str, target: str):
    service = FeatureSelectionService()
    models = ModelService()
    datasets = DatasetService()

    query = {"dataset": dataset, "target": target}
    # Clear feature search results from models
    models.clear_features(query)
    #search_models = models.query_models(query)
    # logging.info("[i] {} models for feature selection".format(len(search_models)))
    # for i, m in enumerate(search_models):
    symbols = datasets.get_dataset_symbols(dataset)
    for i, sym in enumerate(symbols):
        logging.info("==[{}/{}]== Dataset: {} {} {} =====".format(i+1, len(symbols), sym, dataset, target))
        mf = service.create_features_search(target=target, dataset=dataset, symbol=sym, split=0.7, method='importances')
        logging.info("[{}] Start feature search".format(get_timestamp()))
        mf = service.feature_selection(mf, sync=True)
        logging.info("[{}] End feature search".format(get_timestamp()))


if __name__ == '__main__':
    setup_file_logger('feature_selection.log')
    typer.run(main)
