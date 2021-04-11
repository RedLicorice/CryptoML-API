import typer
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.dataset_building_service import DatasetBuildingService
from typing import Optional
from cryptoml_core.logging import setup_file_logger

app = typer.Typer()


@app.command()
def load(bucket: str, filename: str, dataset: str, symbol: str):
    print("Importing {}/{} to {}:{}".format(bucket, filename, dataset, symbol))
    service = DatasetService()
    service.import_from_storage(bucket, filename, dataset, symbol)
    print("Done")


@app.command()
def build(symbol: str, builder: str, ohlcv: str, coinmetrics: str):
    build_args = {
        'ohlcv': ohlcv,
        'coinmetrics': coinmetrics
    }
    print("Building {} [{} -> {}]".format(symbol, build_args, builder))
    service = DatasetBuildingService()
    service.check_builder_args(builder, build_args)
    service.build_dataset(symbol, builder, build_args)
    print("Done")


if __name__ == '__main__':
    setup_file_logger('create_models.log')
    app()
