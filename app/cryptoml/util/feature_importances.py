def label_feature_importances(fit_estimator, labels):
    feature_importances = None
    if hasattr(fit_estimator, 'feature_importances_'):
        feature_importances = fit_estimator.feature_importances_
    elif hasattr(fit_estimator, 'named_steps') and hasattr(fit_estimator.named_steps, 'c') and \
            hasattr(fit_estimator.named_steps.c, 'feature_importances_'):
        feature_importances = fit_estimator.named_steps.c.feature_importances_
    if feature_importances is not None:
        importances = {labels[i]: v for i, v in enumerate(feature_importances)}
        labeled = {str(k): float(v) for k, v in sorted(importances.items(), key=lambda item: -item[1])}
        return labeled
    return None


def label_rank(rank, labels):
    pass


def label_support(support, labels):
    return [c for i, c in enumerate(labels) if support[i]]
