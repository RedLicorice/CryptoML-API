import pandas as pd
from joblib import Parallel, delayed

from cryptoml_core.exceptions import MessageException


def _test_window(est, parameters, X, y, e):
    # Get data for this window
    train_X = X.iloc[:-1, :]
    train_y = y.iloc[:-1]
    # test_X = X.iloc[-1, :].values.reshape(1, -1) # Test contains a single sample so need to reshape
    test_X = X.iloc[-1:, :] # This way it returns a pandas dataframe with a single row
    test_y = y.iloc[-1]

    est.set_params(**parameters)
    est = est.fit(train_X, train_y)
    pred = est.predict(test_X)
    proba = est.predict_proba(test_X)
    result = {
        'time': e,
        'predicted': pred[0],
        'label': test_y
    }
    for cls, prob in enumerate(proba[0]):
        result['predicted_proba_'+str(cls)] = prob
    return result


def test_windows(est, parameters, X, y, ranges, parallel=True):
    if parallel:
        results = Parallel(n_jobs=-1)(delayed(_test_window)(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges)
    else:
        results = [_test_window(est, parameters, X.loc[b:e, :], y.loc[b:e], e) for b, e in ranges]
    df = pd.DataFrame(results)
    if df.empty:
        raise MessageException("TestWindows: Empty result dataframe!")
    df = df.set_index('time')
    return df
