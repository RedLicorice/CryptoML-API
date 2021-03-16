from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__weights': ['distance', 'uniform'],  # Number of neighbors to use by default for kneighbors queries.
    'c__n_neighbors': [5, 4, 3],  # Number of neighbors to use by default for kneighbors queries.
    'c__leaf_size': [30, 100, 300],  # Number of neighbors to use by default for kneighbors queries.
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy='mean')),
    ('s', StandardScaler()),
    ('o', SMOTE()),
    ('c', KNeighborsClassifier()),
])
