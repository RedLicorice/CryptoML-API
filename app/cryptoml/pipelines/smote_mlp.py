from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__hidden_layer_sizes': [(2, 4), (4, 8), (20, 20), (100, 100)],
    'c__solver': ['adam'],
    'c__activation': ['logistic', 'relu'],
    # 'c__alpha': [0.0001, 0.001, 0.01],
    # 'c__learning_rate': ['constant', 'adaptive'],  # Only used for SGD
    'c__random_state': [0],
    'c__max_iter': [1000]
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy='mean')),
    ('s', StandardScaler()),
    ('o', SMOTE()),
    ('c', MLPClassifier()),
])
