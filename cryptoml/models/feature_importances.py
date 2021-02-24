from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, XGBClassifier
from cryptoml.util.feature_importances import label_feature_importances

def get_feature_importances():
    estimator = Pipeline([
        ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
        ('s', StandardScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
        ('c', XGBClassifier(use_label_encoder=False)),
    ])
    return None