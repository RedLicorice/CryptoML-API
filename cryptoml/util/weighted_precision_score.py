from sklearn.metrics import make_scorer, precision_score, classification_report
import numpy as np


def weighted_macro_precision_score(y_true, y_pred, **kwargs):
    unique, _, _, counts = np.unique(y_true, return_counts=True)
    # If classification is binary, fallback to default precision
    if unique.shape[0] <= 2:
        return precision_score(y_true, y_pred, **kwargs)
    # Get separate precision, recall and f-score for all classes
    report = classification_report(y_true, y_pred, return_dict=True, **kwargs)
    # Get a weighted macro average
    weights = kwargs.get('weights', {label: 1.0 for label in unique})
    result = 0.00
    for label in unique:
        result += report[label]['precision'] * weights[label]
    result /= max(np.sum(weights.values()), 1) # We expect to divide a weighted average by the sum of the weights
    return result

def get_scorer(weights={0:1.0, 1:1.0, 2:1.0}, **kwargs):
    return make_scorer(weighted_macro_precision_score, weights=weights, **kwargs)