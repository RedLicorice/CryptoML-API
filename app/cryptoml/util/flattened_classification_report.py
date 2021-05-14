from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from imblearn.metrics import classification_report_imbalanced
import numpy as np
import pandas as pd
from cryptoml_core.exceptions import MessageException


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
        if str(k).isnumeric():
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
    return {k: float(v) for k, v in result.items()}


def roc_auc_report(y_true, y_pred, y_pred_proba):
    if np.isnan(y_true.values).any() or np.isinf(y_true.values).any():
        raise MessageException("y_true contains NaN")
    if np.isnan(y_pred_proba.values).any() or np.isinf(y_pred_proba.values).any():
        # If classifier has diverged, predict_proba will contain nans.
        # We replace them with 0
        with pd.option_context('mode.use_inf_as_na', True):
            y_pred_proba = y_pred_proba.fillna(value=0)
        #raise MessageException("y_pred_proba contains NaN")
    classes = np.unique(y_true)
    result = {}
    if classes.size < 2:
        result['roc_auc_ovo_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovo')
        result['roc_auc_ovo_weighted'] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovo')
        result['roc_auc_ovr_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
        result['roc_auc_ovr_weighted'] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
    else:
        result['roc_auc'] = roc_auc_score(y_true, y_pred)
    # fpr_0, tpr_0, thr_0 = roc_curve(y_true, y_pred, pos_label=0)
    # fpr_1, tpr_1, thr_1 = roc_curve(y_true, y_pred, pos_label=1)
    # fpr_2, tpr_2, thr_2 = roc_curve(y_true, y_pred, pos_label=2)

    return {k: float(v) for k, v in result.items()}
