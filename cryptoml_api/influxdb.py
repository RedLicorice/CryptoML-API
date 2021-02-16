from influxdb import DataFrameClient
from .config import config


dataframe_client: DataFrameClient = None

def get_dataframe_client() -> DataFrameClient:
    global dataframe_client
    if not dataframe_client:
        dataframe_client = DataFrameClient(
            host=config['database']['influxdb']['host'].get(str),
            port=config['database']['influxdb']['port'].get(int),
            username=config['database']['influxdb']['username'].get(str),
            password=config['database']['influxdb']['password'].get(str),
            database=config['database']['influxdb']['database'].get(str)
        )
    return dataframe_client
