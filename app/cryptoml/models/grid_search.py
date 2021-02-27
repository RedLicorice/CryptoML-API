from cryptoml.util.import_proxy import GridSearchCV
from sklearn.utils import parallel_backend
from cryptoml_core.deps.dask import get_client
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
    dask = get_client()
    logging.info("GridSearch: X_type {} X_shape {} y_type {} y_shape {}".format(type(X_train), X_train.shape, type(y_train), y_train.shape))
    if kwargs.get('sync', False):
        gscv.fit(X_train, y_train)
    else:
        with parallel_backend('dask'):
            gscv.fit(X_train, y_train)
    return gscv