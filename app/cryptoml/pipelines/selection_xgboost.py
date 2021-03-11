from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, MinMaxScaler, XGBClassifier

PARAMETER_GRID = {}

PARAMETERS = {
    "colsample_bylevel": 1,
    "colsample_bynode": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.001,
    "max_depth": 2,
    "n_estimators": 500,
    "num_parallel_tree": 1,
    "objective": "binary:logistic",
    "reg_alpha": 0,
    "reg_lambda": 1,
    "subsample": 1,
    "use_label_encoder": False,
    "seed": None,
    "random_state": 0
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy="mean")),  # Replace nan's with the mean value between previous and next observation
    ('s', StandardScaler()),  # Standardize data so that Mean and StdDev are < 1
    ('n', MinMaxScaler(feature_range=(0, 1))),  # Normalize data in range [0, 1]
    ('c', XGBClassifier(**PARAMETERS)),
])

