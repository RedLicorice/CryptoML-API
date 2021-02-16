#from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
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

    #with joblib.parallel_backend('dask'):
    gscv = GridSearchCV(
        estimator=est,
        param_grid=parameters,
        cv=cv,
        #n_jobs=n_jobs,
        scoring=scoring
    )
    gscv.fit(X_train, y_train)
    return gscv