from cryptoml.util.selection_pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler
from scipy.stats import randint, uniform, loguniform


PARAMETER_GRID = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8],  # Number of samples per estimator in the ensemble
    'max_features': [0.2, 0.8, 1.0],  # Number of features per estimator in the ensemble
    'base_estimator__c__criterion': ['gini'],  #, 'entropy'],
    'base_estimator__c__splitter': ['random', 'best'],  # 'best',
    'base_estimator__c__max_depth': [2, 3],  #None,
    'base_estimator__c__min_samples_split': [2],
    'base_estimator__c__min_samples_leaf': [1, 0.05, 0.2],
    'base_estimator__c__min_weight_fraction_leaf': [0.0],  # 0.01, 0.1],
    'base_estimator__c__max_features': ['auto'],  #'sqrt',,  'log2'
    'base_estimator__c__class_weight': [None, 'balanced']
}

PARAMETER_DISTRIBUTION = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': uniform(0.5, 0.8),  # Number of samples per estimator in the ensemble
    'max_features': uniform(0.2, 1.0),  # Number of features per estimator in the ensemble
    'base_estimator__c__criterion': ['gini'],  #, 'entropy'],
    'base_estimator__c__splitter': ['random', 'best'],  # 'best',
    'base_estimator__c__max_depth': randint(2, 12),  #None,
    'base_estimator__c__min_samples_split': [2],
    'base_estimator__c__min_samples_leaf': uniform(0.05, 1),
    'base_estimator__c__min_weight_fraction_leaf': [0.0],  # 0.01, 0.1],
    'base_estimator__c__max_features': ['auto'],  #'sqrt',,  'log2'
    'base_estimator__c__class_weight': [None, 'balanced']
}

pipeline = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', DecisionTreeClassifier()),
])

estimator = BaggingClassifier(base_estimator=pipeline)