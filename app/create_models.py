import typer
from cryptoml_core.services.model_service import ModelService
from typing import Optional
from cryptoml_core.logging import setup_file_logger


def main(dataset: str, pipeline: str):
    models = ModelService()
    query = {'type': "FEATURES"}
    if pipeline != 'all':
        query['pipeline'] = pipeline
    if dataset != 'all':
        query['name'] = dataset
    models.create_classification_models(query=query)


if __name__ == '__main__':
    setup_file_logger('create_models.log')
    typer.run(main)
