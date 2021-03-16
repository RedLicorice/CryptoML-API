from cryptoml.util.selection_pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler, StandardScaler

PARAMETER_GRID = {
    'c__weights': ['distance', 'uniform'],  # Number of neighbors to use by default for kneighbors queries.
    'c__n_neighbors': [15, 10, 5, 3],  # Number of neighbors to use by default for kneighbors queries.
    'c__leaf_size': [30, 100, 300],  # Number of neighbors to use by default for kneighbors queries.
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    # ('n', MinMaxScaler()),
    ('c', KNeighborsClassifier()),
])