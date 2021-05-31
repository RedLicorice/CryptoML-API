import typer
from cryptoml_core.services.feature_selection import FeatureSelectionService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.logging import setup_file_logger
import logging
from typing import Optional


def main(dataset: str, target: str, method: str, split: Optional[float] = 0.7, replace: Optional[bool] = False):
    service = FeatureSelectionService()

    symbols = service.get_available_symbols(dataset)
    for i, sym in enumerate(symbols):
        logging.info("==[{}/{}]== Dataset: {} {} {} =====".format(i+1, len(symbols), sym, dataset, target))
        logging.info("[{}] Start feature search".format(get_timestamp()))
        mf = service.feature_selection_new(
            symbol=sym,
            dataset=dataset,
            target=target,
            split=split,
            method=method,
            replace=replace
        )
        logging.info("[{}] End feature search".format(get_timestamp()))


if __name__ == '__main__':
    setup_file_logger('feature_selection_new.log')
    typer.run(main)
