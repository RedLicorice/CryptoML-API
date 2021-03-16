from cryptoml.util.selection_pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler


PARAMETER_GRID = {
    'c__n_estimators': [50, 250, 500], # Number of estimators to use in ensemble
    'c__learning_rate': [0.1, 0.01, 0.001],
    # 'c__base_estimator__criterion': ['gini', 'entropy'],
    'c__base_estimator__splitter': ['best'],
    'c__base_estimator__max_depth': [None, 3],
    #'c__base_estimator__min_samples_split': [2],
    'c__base_estimator__min_samples_leaf': [1, 2, 3],
    #'c__base_estimator__min_weight_fraction_leaf': [0.0],
    #'c__base_estimator__max_features': ['auto'],
    'c__base_estimator__class_weight': ['balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    ('c', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())),
])