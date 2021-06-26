import pandas as pd
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
from cryptoml_core.exceptions import MessageException
import numpy as np
import logging
from cryptoml.util.shap import get_shap_values

def _test_window(est, parameters, X, y, e):
    # Get data for this window
    train_X = X.iloc[:-1, :]
    train_y = y.iloc[:-1]
    # test_X = X.iloc[-1, :].values.reshape(1, -1) # Test contains a single sample so need to reshape
    test_X = X.iloc[-1:, :] # This way it returns a pandas dataframe with a single row
    test_y = y.iloc[-1]

    y_unique, _, y_counts = np.unique(train_y, return_index=True, return_counts=True)
    if (y_counts < 3).any():
        logging.warning(f"train_y contains less than 3 samples for some class! \nUnique: {y_unique}\nCounts: {y_counts}")
        return None
    est.set_params(**parameters)
    start_at = datetime.utcnow().timestamp()
    est = est.fit(train_X, train_y)
    dur = datetime.utcnow().timestamp() - start_at
    pred = est.predict(test_X)
    proba = est.predict_proba(test_X)
    shap_per_class, shap_expected_values = get_shap_values(model=est, X=test_X, X_train=train_X, bytes=False)
    result = {
        'time': e,
        'duration': dur,
        'predicted': pred[0],
        'label': test_y
    }
    if proba.any():
        for cls, prob in enumerate(proba[0]):
            result['predicted_proba_'+str(cls)] = prob
    for u, c in zip(y_unique, y_counts):
        result[f"class_{u}_count"] = c

    return {
        "entry": result,
        "shap": {
            "timestamp": e,
            "values": shap_per_class,
            "expected_values": shap_expected_values
        }
    }

def _train_window(est, parameters, X_train, y_train):
    y_unique, _, y_counts = np.unique(y_train, return_index=True, return_counts=True)
    if (y_counts < 3).any():
        logging.warning(f"train_y contains less than 3 samples for some class! \nUnique: {y_unique}\nCounts: {y_counts}")
        return None
    est.set_params(**parameters)
    start_at = datetime.utcnow().timestamp()
    est = est.fit(X_train, y_train)
    dur = datetime.utcnow().timestamp() - start_at
    est.fit_time = dur
    return est

def _predict_window(est, X_test, y_test, e):
    if not est.is_fit:
        logging.exception(f"Predict window needs a fit estimator!")
    pred = est.predict(X_test)
    proba = est.predict_proba(X_test)
    result = {
        'time': e,
        'duration': est.fit_time,
        'predicted': pred[0],
        'label': y_test
    }
    y_unique, _, y_counts = np.unique(est.train_y, return_index=True, return_counts=True)
    if proba.any():
        for cls, prob in enumerate(proba[0]):
            result['predicted_proba_'+str(cls)] = prob
    for u, c in zip(y_unique, y_counts):
        result[f"class_{u}_count"] = c
    return result

def _shap_window(est, X, X_train, y, e):
    shap_per_class, shap_expected_value = get_shap_values(model=est, X=X, X_train=X_train, bytes=False)
    result = {
        'time': e,
        'expected': y,
        'shap_expected': shap_expected_value,
    }

def test_windows(est, parameters, X, y, ranges, parallel=True, **kwargs):
    _n_jobs = int(kwargs.get('n_jobs', cpu_count() / 2))
    if parallel:
        results = Parallel(n_jobs=_n_jobs)(delayed(_test_window)(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges)
    else:
        results = [_test_window(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges]
    results_data = [r["entry"] for r in results if r is not None]
    df = pd.DataFrame(results_data)
    if df.empty:
        raise MessageException("TestWindows: Empty result dataframe!")
    df = df.set_index('time')

    shap_df_per_class = {}
    for shap in [r["shap"] for r in results if r is not None]:
        for cls, arr in enumerate(shap["values"]):
            print(arr)
    # shap_df = pd.concat(shap_dfs, axis='columns')
    # shap_df = shap_df.reindex(sorted(shap_df.columns), axis=1)

    return df


def predict_day(est, parameters, X, y, day, **kwargs):
    result = _test_window(est, parameters, X, y, day)
    df = pd.DataFrame(result["entry"])
    if df.empty:
        raise MessageException("predict_day: Empty result dataframe!")
    df = df.set_index('time')
    return df