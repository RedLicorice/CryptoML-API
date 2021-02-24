from sklearn.base import BaseEstimator
import dask.dataframe as dd
import pandas as pd

class DaskDataframeConverter(BaseEstimator):
    def fit(self, X, y=None):
        if(isinstance(X, pd.DataFrame)):
            X = dd.DataFrame(X)
        return self

