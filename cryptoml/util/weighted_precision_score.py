from sklearn.metrics import make_scorer, precision_score, classification_report
import numpy as np


def weighted_macro_precision_score(y_true, y_pred, **kwargs):
    unique, counts = np.unique(y_true, return_counts=True)
    # If classification is binary, fallback to default precision
    if unique.shape[0] <= 2:
        return precision_score(y_true, y_pred, **kwargs)
    # Get separate precision, recall and f-score for all classes
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # Get a weighted macro average
    weights = kwargs.get('weights')
    if not weights:
        weights={label: 1.0 for label in unique}
    result = 0.00
    total_weight = np.sum([v for k,v in weights.items()])
    for label in unique:
        result += report[str(label)]['precision'] * weights[str(label)]
    result /= max(total_weight, 1) # We expect to divide a weighted average by the sum of the weights
    return result

def get_weighted_precision_scorer(weights={0:1.0, 1:1.0, 2:1.0}, **kwargs):
    return make_scorer(weighted_macro_precision_score, weights=weights, **kwargs)