from cryptoml.selection_pipeline import Pipeline
from sklearn.svm import LinearSVC
from dask_ml.preprocessing import StandardScaler
from dask_ml.impute import SimpleImputer


PARAMETER_GRID = {
    'c__penalty': ['l2'], # Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC.
    'c__loss': ['squared_hinge'], #  ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
    'c__C': [0.1, 0.5, 1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__class_weight':[None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', LinearSVC()),
])