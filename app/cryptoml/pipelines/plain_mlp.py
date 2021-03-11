from cryptoml.util.selection_pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from cryptoml.util.import_proxy import SimpleImputer, MinMaxScaler

PARAMETER_GRID = {
    'c__hidden_layer_sizes':[(2,4), (4,8)],
    'c__solver':['adam'],
    'c__activation':['logistic','tanh','relu'],
    'c__alpha':[0.0001, 0.001, 0.01],
    'c__learning_rate':['constant','adaptive'],
    'c__random_state':[0],
    'c__max_iter':[2000]
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', MinMaxScaler()),
    ('c', MLPClassifier()),
])