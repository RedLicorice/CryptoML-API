import typer
from cryptoml_core.services.model_service import ModelService
import json, time


# app = typer.Typer()


# @app.command()
# def clear_parameters(queryfile: str):
#     with open(queryfile, 'r') as f:
#         query = json.load(f)
#     service = ModelService()
#     service.clear_parameters(query)
#
#
# @app.command()
# def clear_features(queryfile: str):
#     with open(queryfile, 'r') as f:
#         query = json.load(f)
#     service = ModelService()
#     service.clear_features(query)


# @app.command()
def main(queryfile: str):
    with open(queryfile, 'r') as f:
        query = json.load(f)
    service = ModelService()
    print(f"Clearing {query}")
    time.sleep(5)
    service.clear_tests(query)


if __name__ == '__main__':
    # app()
    typer.run(main)
