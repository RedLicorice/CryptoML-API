import typer
from cryptoml_core.services.model_service import ModelService
from typing import Optional


def main(dataset: str, target: str, pipeline: str, features: Optional[str] = None, parameters: Optional[str] = None):
    models = ModelService()
    query = {}
    if pipeline != 'all':
        query['pipeline'] = pipeline
    if target != 'all':
        query['target'] = target
    if dataset != 'all':
        query['dataset'] = dataset
    models.create_classification_models(query=query)


if __name__ == '__main__':
    typer.run(main)
