from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced


def flattened_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    result = {}
    for k, v in report.items():
        if k.isnumeric():
            for m, _v in v.items():
                result['{}_{}'.format(m, k)] = _v
        elif k.endswith('avg'):
            splits = k.split(' ')
            for m, _v in v.items():
                if m == 'support':
                    continue
                result['{}_{}_avg'.format(m, splits[0])] = _v

        else:
            result[k] = v
    return result


def flattened_classification_report_imbalanced(y_true, y_pred):
    report = classification_report_imbalanced(y_true, y_pred, zero_division=0, output_dict=True)
    result = {}
    for k, v in report.items():
        if k.isnumeric():
            for m, _v in v.items():
                result['{}_{}'.format(m, k)] = _v
        elif k.endswith('avg'):
            splits = k.split(' ')
            for m, _v in v.items():
                if m == 'support':
                    continue
                result['{}_{}_avg'.format(m, splits[0])] = _v
        else:
            result[k] = v
    return result
