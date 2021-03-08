from cryptoml.util.import_proxy import GridSearchCV
from sklearn.utils import parallel_backend
import logging

# Perform Cross-Validated grid search with a given estimator over a given
# parameter grid, return search results
def grid_search(est, parameters, X_train, y_train, **kwargs):
    cv = kwargs.get('cv', 5)
    scoring = kwargs.get('scoring', 'accuracy')

    gscv = GridSearchCV(
        estimator=est,
        param_grid=parameters,
        cv=cv,
        scoring=scoring
    )
    if kwargs.get('sync', False):
        logging.info("GridSearch (SYNC): X_type {} X_shape {} y_type {} y_shape {}"\
                     .format(type(X_train), X_train.shape, type(y_train), y_train.shape))
        gscv.fit(X_train, y_train)
    else:
        with parallel_backend('dask'):
            logging.info("GridSearch (DASK): X_type {} X_shape {} y_type {} y_shape {}" \
                         .format(type(X_train), X_train.shape, type(y_train), y_train.shape))
            gscv.fit(X_train, y_train)
    return gscv

def parameter_search(est, parameters, X_train, y_train, **kwargs):
    cv = kwargs.get('cv', 5)
    scoring = kwargs.get('scoring', 'accuracy')

    gscv = GridSearchCV(
        estimator=est,
        param_grid=parameters,
        cv=cv,
        scoring=scoring
    )
    if kwargs.get('sync', False):
        logging.info("GridSearch (SYNC): X_type {} X_shape {} y_type {} y_shape {}"\
                     .format(type(X_train), X_train.shape, type(y_train), y_train.shape))
        gscv.fit(X_train, y_train)
    else:
        with parallel_backend('dask'):
            logging.info("GridSearch (DASK): X_type {} X_shape {} y_type {} y_shape {}" \
                         .format(type(X_train), X_train.shape, type(y_train), y_train.shape))
            gscv.fit(X_train, y_train)
    return gscv