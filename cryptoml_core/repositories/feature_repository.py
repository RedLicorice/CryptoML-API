import pandas as pd
from cryptoml_core.deps.influxdb.client import DataFrameClient
from cryptoml_core.deps.influxdb.queries import query_dataframe
from cryptoml_core.deps.influxdb.distributed_queries import query_dask_dataframe


class FeatureRepository:
    def __init__(self):
        pass

    def delete_dataset(self, dataset: str):
        measure = 'dataset_{}'.format(dataset)
        with DataFrameClient() as client:
            return client.query("DROP MEASUREMENT {}".format(measure))

    def delete_features(self, dataset: str, symbol: str):
        measure = 'dataset_{}'.format(dataset)
        with DataFrameClient() as client:
            return client.query("DELETE FROM {} WHERE symbol='{}'".format(measure, symbol))

    def put_features(self, df: pd.DataFrame, dataset: str, symbol: str, facet: str = 'default'):
        measure = 'dataset_{}'.format(dataset)
        with DataFrameClient() as client:
            client.write_points(df, measure, {'symbol':symbol, 'facet': facet})
        return measure

    # def put_target(self, s: pd.Series, type: str, symbol: str):
    #     measure = 'target_{}'.format(type)
    #     df = pd.DataFrame()
    #     df['label'] = s
    #     with DataFrameClient() as client:
    #         client.write_points(df, measure, {'symbol':symbol})
    #     return measure

    def get_features(self, dataset, symbol, **kwargs):
        measure = 'dataset_{}'.format(dataset)
        df = query_dataframe(measure, tags={'symbol': symbol, 'facet': kwargs.get('facet')}, **kwargs)
        return df

    def get_target(self, type, symbol, **kwargs):
        measure = 'dataset_target'
        df = query_dataframe(measure, tags={'symbol': symbol}, columns=[type], **kwargs)
        s = df[type]
        s.name = 'label'
        return s

    def get_dask_features(self, dataset, symbol, **kwargs):
        measure = 'dataset_{}'.format(dataset)
        df = query_dask_dataframe(measure, tags={'symbol': symbol, 'facet':kwargs.get('facet')}, delta={'days':10000})
        return df

    def get_dask_target(self, type, symbol, **kwargs):
        measure = 'target_{}'.format(type)
        df = query_dask_dataframe(measure, tags={'symbol': symbol}, delta={'days': 10000})
        return df['label']

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
