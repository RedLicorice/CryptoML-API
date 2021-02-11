from cryptoml.util.selection_pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from dask_ml.preprocessing import StandardScaler
from dask_ml.impute import SimpleImputer


PARAMETER_GRID = {
    'c__n_neighbors':[5, 4, 3], # Number of neighbors to use by default for kneighbors queries.
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    ('c', KNeighborsClassifier()),
])