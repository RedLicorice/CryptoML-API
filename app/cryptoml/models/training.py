import numpy as np
import pandas as pd
from sklearn.utils import parallel_backend

def train_model(est, parameters, X_train: pd.DataFrame, y_train: pd.Series):
    _est = est.set_params(**parameters)
    _est = _est.fit(X_train, y_train)
    return _est

def predict_model(est, X_pred: np.array):
    return est.predict(X_pred)