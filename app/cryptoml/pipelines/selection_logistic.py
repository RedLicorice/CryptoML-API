from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

PARAMETER_GRID = {}

PARAMETERS = {
    "C": 1.0,
    "penalty": "l2",
    "dual": False,
    "class_weight": "balanced",
    "solver": "sag",
    "max_iter": 500
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy='mean')),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),  # Standardize data so that Mean and StdDev are < 1
    ('n', MinMaxScaler(feature_range=(-1, 1))),  # Normalize data in range [0, 1]
    ('c', LogisticRegression(**PARAMETERS)),
])
