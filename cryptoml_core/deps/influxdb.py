from influxdb import DataFrameClient
from cryptoml_core.deps.config import config
from cryptoml_core.util.ident import get_ident

dataframe_client: DataFrameClient = None
ident: str = None

def get_dataframe_client() -> DataFrameClient:
    global dataframe_client, ident
    cur_ident = get_ident()
    if not dataframe_client or cur_ident != ident:
        dataframe_client = DataFrameClient(
            host=config['database']['influxdb']['host'].get(str),
            port=config['database']['influxdb']['port'].get(int),
            username=config['database']['influxdb']['username'].get(str),
            password=config['database']['influxdb']['password'].get(str),
            database=config['database']['influxdb']['database'].get(str)
        )
        ident = cur_ident
    return dataframe_client
