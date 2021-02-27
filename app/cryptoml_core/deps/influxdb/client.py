from influxdb import DataFrameClient as InfluxDBDataFrameClient
from influxdb import InfluxDBClient
from cryptoml_core.deps.config import config


db_config = {
    'host': config['database']['influxdb']['host'].get(str),
    'port': config['database']['influxdb']['port'].get(int),
    'username': config['database']['influxdb']['username'].get(str),
    'password': config['database']['influxdb']['password'].get(str),
    'database': config['database']['influxdb']['database'].get(str),
    'use_udp': config['database']['influxdb']['udp'].get(bool),
    'udp_port': config['database']['influxdb']['udp'].get(int)
}


# def get_dataframe_client() -> DataFrameClient:
#     client = DataFrameClient(**db_config)
#     return client
#
#
# def get_dict_client() -> InfluxDBClient:
#     client = InfluxDBClient(**db_config)
#     return client

class DictClient(InfluxDBClient):
    def __init__(self):
        super(DictClient, self).__init__(**db_config)

    def __enter__(self):
        super(DictClient, self).__enter__()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        super(DictClient, self).__exit__(_exc_type, _exc_value, _traceback)


class DataFrameClient(InfluxDBDataFrameClient):
    def __init__(self):
        super(DataFrameClient, self).__init__(**db_config)

    def __enter__(self):
        super(DataFrameClient, self).__enter__()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        super(DataFrameClient, self).__exit__(_exc_type, _exc_value, _traceback)
