import typer
from cryptoml_core.services.model_service import ModelService
import json


app = typer.Typer()


@app.command()
def clear(queryfile: str):
    with open(queryfile, 'r') as f:
        query = json.load(f)
    service = ModelService()
    service.clear_parameters(query)
    service.clear_tests(query)


@app.command()
def clear_features(queryfile: str):
    with open(queryfile, 'r') as f:
        query = json.load(f)
    service = ModelService()
    service.clear_features(query)


if __name__ == '__main__':
    app()
