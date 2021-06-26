import typer
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.exceptions import MessageException
import json
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging


def main(pipeline: str, dataset: str, symbol: str, target: str, window: int, features: Optional[str] = None, parameters: Optional[str] = None, save: Optional[bool] = True):
    models = ModelService()

    try:
        logging.info(f"[{get_timestamp()}] {pipeline}({dataset}.{symbol}, W={window}) -> {target},"
                     f"Features: {features}, Parameters: {parameters}, {'Persisted' if save else 'Non-Persisted'}")
        models.test_model_new(
            pipeline=pipeline,
            dataset=dataset,
            symbol=symbol,
            target=target,
            features=features,
            parameters=parameters,
            split=0.7,
            window={'days': window},
            save=save
        )
    except MessageException as e:
        logging.error("[!] TEST FAILED, MESSAGE: " + e.message)
        pass
    except Exception as e:
        logging.exception("[!] TEST FAILED: " + str(e))
        pass
    else:
        logging.info(f"[{get_timestamp()}] {pipeline}({dataset}.{symbol}, W={window}) -> {target} DONE")


if __name__ == '__main__':
    setup_file_logger('test_model_new.log')
    typer.run(main)
