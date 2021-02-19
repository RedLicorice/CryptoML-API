from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend
from dask.distributed import Client

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
    dask = Client('dask-scheduler:8786')
    with parallel_backend('dask'):
        gscv.fit(X_train, y_train)
    return gscv