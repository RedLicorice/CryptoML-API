from influxdb import DataFrameClient
import pandas as pd
import numpy as np
from cryptoml_api.config import config

class FeatureRepository:
    def __init__(self):
        self.client: DataFrameClient = DataFrameClient(
            host=config['database']['influxdb']['host'].get(str),
            port=config['database']['influxdb']['port'].get(int),
            username=config['database']['influxdb']['username'].get(str),
            password=config['database']['influxdb']['password'].get(str),
            database=config['database']['influxdb']['database'].get(str)
        )

    def put_features(self, df: pd.DataFrame, dataset: str, symbol: str):
        # Tags are indexed
        self.client.write_points(df, 'features', {'dataset':dataset, 'symbol':symbol})

    def get_features(self, dataset, symbol, **kwargs):
        query = "SELECT * FROM features WHERE dataset='{}' AND symbol='{}'".format(dataset, symbol)
        res = self.client.query(query)
        df = res['features']
        df = df.drop(labels=['dataset', 'symbol'], axis='columns')
        return df

    def put_target(self, s: pd.Series, type: str, symbol: str):
        df = pd.DataFrame()
        df['value'] = s
        self.client.write_points(df, 'targets', {'type':type, 'symbol':symbol})

    def get_target(self, type, symbol, **kwargs):
        query = "SELECT value FROM targets WHERE type='{}' AND symbol='{}'".format(type, symbol)
        res = self.client.query(query)
        df = res['targets']
        return df['value']

if __name__ == '__main__':
    print("Open Dataframe on disk")
    df = pd.read_csv('B:\\Tesi-POLITO\\LSTM_forecaster\\data\\datasets\\timedspline\\csv\\BTC.csv', parse_dates=True, index_col='Date')
    dft = pd.read_csv('B:\\Tesi-POLITO\\LSTM_forecaster\\data\\datasets\\timedspline\\csv\\BTC_target.csv', parse_dates=True, index_col='Date')
    print("Instantiate Repository")
    repo = FeatureRepository()
    print("Import features to repo")
    repo.put_features(df, 'timedspline', 'BTC')
    print("Import target to repo")
    repo.put_target(dft['class'], 'class', 'BTC' )
    print("Get features from repo")
    features = repo.get_features('timedspline', 'BTC')
    print("Get target from repo")
    target = repo.get_target('class', 'BTC')
    print(features.head())
    print(target.head())