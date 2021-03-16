from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, MinMaxScaler, XGBClassifier

PARAMETER_GRID = {}

PARAMETERS = {
    "colsample_bylevel": 0.8,
    "colsample_bynode": 1,
    "colsample_bytree": 0.8,
    "learning_rate": 0.3,
    "max_depth": 6,
    "n_estimators": 500,
    "num_parallel_tree": 1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "subsample": 1,
    "use_label_encoder": False,
    "seed": None,
    "random_state": 0,
    "objective": "multi:softmax",
    "eval_metric": "mlogloss"
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy="mean")),  # Replace nan's with the mean value between previous and next observation
    ('s', StandardScaler()),  # Standardize data so that Mean and StdDev are < 1
    ('c', XGBClassifier(**PARAMETERS)),
])

