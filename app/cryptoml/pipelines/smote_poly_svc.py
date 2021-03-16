from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__C': [1, 5, 10],
    # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__kernel': ['poly'],
    'c__gamma': ['scale', 'auto'],
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'c__degree': [2, 3, 4],
    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'c__class_weight': [None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy='mean')),
    ('s', StandardScaler()),
    ('o', SMOTE()),
    ('c', SVC()),
])
