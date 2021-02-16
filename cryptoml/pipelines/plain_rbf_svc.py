from cryptoml.util.selection_pipeline import Pipeline
from sklearn.svm import SVC
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler


PARAMETER_GRID = {
    'c__C': [1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__kernel': ['rbf'],
    'c__gamma': ['scale', 'auto'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'c__class_weight':[None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', SVC()),
])