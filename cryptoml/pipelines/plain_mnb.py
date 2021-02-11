from cryptoml.util.selection_pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from dask_ml.preprocessing import StandardScaler
from dask_ml.impute import SimpleImputer


PARAMETER_GRID = {
    'c__alpha':[0.2, 0.4, 0.6, 0.8, 1], # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    ('c', MultinomialNB()),
])