import pandas as pd
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
from cryptoml_core.exceptions import MessageException
import numpy as np
import logging

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
    return result


def test_windows(est, parameters, X, y, ranges, parallel=True, **kwargs):
    _n_jobs = int(kwargs.get('n_jobs', cpu_count() / 2))
    if parallel:
        results = Parallel(n_jobs=_n_jobs)(delayed(_test_window)(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges)
    else:
        results = [_test_window(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges]
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    if df.empty:
        raise MessageException("TestWindows: Empty result dataframe!")
    df = df.set_index('time')
    return df


def predict_day(est, parameters, X, y, day, **kwargs):
    result = _test_window(est, parameters, X, y, day)
    df = pd.DataFrame([result])
    if df.empty:
        raise MessageException("predict_day: Empty result dataframe!")
    df = df.set_index('time')
    return df