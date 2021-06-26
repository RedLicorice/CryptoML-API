import typer

from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.model_service import ModelService
from typing import Optional
from cryptoml_core.logging import setup_file_logger, logging


def main():
    models = ModelService()
    datasets = DatasetService()
    query = {
        "dataset": "merged_new",
        "target": "class"
    }
    all_models = models.query_models(query=query)
    for m in all_models:
        ds = datasets.get_dataset(name=m.dataset, symbol=m.symbol)
        fs = DatasetService.get_feature_selection(ds=ds, method='importances_shap', target=m.target)
        if not fs:
            logging.error(f"Dataset {m.dataset}{m.symbol} -> {m.target} does not have feature selection")
            continue

        if not m.parameters:
            logging.error(f"Model {m.pipeline}({m.dataset}{m.symbol}) -> {m.target} does not have parameters")
            continue

        for mp in m.parameters:
            count = 0
            for f in mp.features:
                if not f in fs.features:
                    logging.error(f"Model {m.pipeline}({m.dataset}{m.symbol}) -> {m.target} parameter search done without fixing features!")
                else:
                    count += 1
            logging.info(f"Model {m.pipeline}({m.dataset}{m.symbol}) -> {m.target} GRIDSEARCH {mp.parameter_search_method} done with {count} features")


if __name__ == '__main__':
    setup_file_logger('check_models.log')
    typer.run(main)
