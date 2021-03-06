from cryptoml.util.selection_pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, MinMaxScaler


PARAMETER_GRID = {
    'c__n_estimators': [100, 200, 500],
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__criterion': ['gini'],  # , 'entropy'],
    'c__max_depth': [2, 3, 4],
    'c__min_samples_split': [2],
    'c__min_samples_leaf': [1, 0.05, 0.2],
    'c__max_features': ['auto'],  # 'sqrt',
    'c__class_weight': [None, 'balanced'],#, 'balanced_subsample'
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('c', RandomForestClassifier()),
])