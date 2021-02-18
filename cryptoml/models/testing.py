import numpy as np
import pandas as pd
from .training import train_model, predict_model


def trailing_window_test(
        est, parameters, W, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, **kwargs
    ):
    if X_train.shape[0] <= W:
        raise Exception("Too few training datapoints for window {}".format(W))
    if y_train.shape[0] <= W:
        raise Exception("Too few training labels for window {}".format(W))
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

        _est = train_model(est, parameters, X_train, y_train)
        pred = predict_model(_est, _X_test)

        predictions.append(pred[0])
        labels.append(_y_test[0])
        window += 1
        # print('\t Expect: {} Predict: {}'.format(_y_test[0], pred[0]))

    labels_arr = np.flip(np.array(labels), axis=0)
    predictions_arr = np.flip(np.array(predictions), axis=0)
    return labels_arr, predictions_arr