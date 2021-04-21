from cryptoml.util.selection_pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler, StandardScaler


PARAMETER_GRID = {
    'n_estimators': [5, 10],  # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8, 1.0],  # Number of samples per estimator in the ensemble
    'max_features': [0.5, 1.0],  # Number of features per estimator in the ensemble
    'base_estimator__c__C': [1, 5, 10],  # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'base_estimator__c__kernel': ['poly'],
    'base_estimator__c__gamma': ['scale', 'auto'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'base_estimator__c__degree': [2, 3, 4],  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    # 'base_estimator__c__coef0': [0.0, 0.1, 0.2],  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'base_estimator__c__class_weight': [None, 'balanced']
}

pipeline = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    # ('n', MinMaxScaler()),
    ('c', SVC(probability=True)),
])

estimator = BaggingClassifier(base_estimator=pipeline)
