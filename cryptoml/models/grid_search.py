from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend
from cryptoml_core.deps.dask import get_client

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
    with parallel_backend('dask'):
        gscv.fit(X_train, y_train)
    return gscv