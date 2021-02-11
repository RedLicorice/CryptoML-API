from dask_ml.model_selection import GridSearchCV
import numpy as np
import os
import joblib

# Perform Cross-Validated grid search with a given estimator over a given
# parameter grid, return search results
def grid_search(est, parameters, X_train, y_train, **kwargs):
    cv = kwargs.get('cv', 5)
    expanding_window = kwargs.get('expanding_window', False)
    n_jobs = kwargs.get('n_jobs', 'auto')
    scoring = kwargs.get('scoring', 'accuracy')
    # Parse arguments
    # if n_jobs == 'auto':
    #     n_jobs = os.cpu_count()
    # elif n_jobs.isnumeric():
    #     n_jobs = int(n_jobs)

    with joblib.parallel_backend('dask'):
        gscv = GridSearchCV(
            estimator=est,
            param_grid=parameters,
            cv=cv,
            #n_jobs=n_jobs,
            scoring=scoring
        )
        gscv.fit(X_train, y_train)
        return gscv

def test_model(est, parameters, W, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, **kwargs):
    if X_train.shape[0] <= W:
        raise Exception("Too few training datapoints for window {}".format(W))
    features = np.concatenate((X_train[:-W], X_test))
    target = np.concatenate((y_train[:-W], y_test))

    predictions = []
    labels = []

    # Go in reverse
    window = 1
    for i in range(features.shape[0], 0, -1):
        if i < (W + 1):
            break
        train_start = i - W - 1
        train_end = i - 1
        test_start = i - 1
        test_end = i
        # print('[Window {}]\tTrain: B={} E={}\tTest: B={} E={}'.format(window, train_start, train_end, test_start, test_end))
        _X_train = features[train_start:train_end]
        _y_train = target[train_start:train_end]
        _X_test = features[test_start:test_end]
        _y_test = target[test_start:test_end]

        _est = est.set_params(**parameters)
        _est = _est.fit(_X_train, _y_train)
        pred = _est.predict(_X_test)

        predictions.append(pred[0])
        labels.append(_y_test[0])
        window += 1
        # print('\t Expect: {} Predict: {}'.format(_y_test[0], pred[0]))

    labels_arr = np.flip(np.array(labels), axis=0)
    predictions_arr = np.flip(np.array(predictions), axis=0)
    return (labels_arr, predictions_arr)

def train_model(est, parameters, X_train: np.array, y_train: np.array):
    _est = est.set_params(**parameters)
    _est = _est.fit(X_train, y_train)
    return _est

def predict_model(est, X_pred: np.array):
    return est.predict(X_pred)