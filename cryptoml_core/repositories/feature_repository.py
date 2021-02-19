import pandas as pd
from cryptoml_core.deps.influxdb import get_dataframe_client


class FeatureRepository:
    def __init__(self):
        self.client = get_dataframe_client()

    def put_features(self, df: pd.DataFrame, dataset: str, symbol: str, facet: str = 'default'):
        measure = 'dataset_{}'.format(dataset)
        self.client.write_points(df, measure, {'symbol':symbol, 'facet': facet})

    def get_features(self, dataset, symbol, **kwargs):
        measure = 'dataset_{}'.format(dataset)
        query = "SELECT * FROM {} WHERE symbol='{}'".format(measure, symbol)
        if kwargs.get('begin'):
            query = query + " AND time >= '{}'".format(kwargs.get('begin'))
        if kwargs.get('end'):
            query = query + " AND time < '{}'".format(kwargs.get('end'))
        if kwargs.get('facet'):
            query = query + " AND facet == '{}'".format(kwargs.get('facet'))
        res = self.client.query(query)
        df = res[measure]
        tags = ['symbol', 'facet']
        df = df.drop(labels=[c for c in df.columns if c in tags], axis='columns')
        return df

    def put_target(self, s: pd.Series, type: str, symbol: str):
        measure = 'target_{}'.format(type)
        df = pd.DataFrame()
        df['label'] = s
        self.client.write_points(df, measure, {'symbol':symbol})

    def get_target(self, type, symbol, **kwargs):
        measure = 'target_{}'.format(type)
        query = "SELECT label FROM {} WHERE symbol='{}'".format(measure, symbol)
        if kwargs.get('begin'):
            query = query + " AND time >= '{}'".format(kwargs.get('begin'))
        if kwargs.get('end'):
            query = query + " AND time < '{}'".format(kwargs.get('end'))
        res = self.client.query(query)
        df = res[measure]
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