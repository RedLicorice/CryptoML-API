import shap
import pandas as pd
import numpy as np
import io
import base64
from typing import Optional

class PipelineTree(shap.TreeExplainer):
    def __init__(self, model, data=None, model_output="raw", feature_perturbation="interventional", feature_names=None,
                 **deprecated_options):
        model_repr = str(model)
        self.__original_pipeline = model
        if 'Pipeline' in model_repr and hasattr(model, 'named_steps') and 'c' in model.named_steps:
            model = model.named_steps['c']
        super(shap.TreeExplainer, self).__init__(model=model, data=data, model_output=model_output, feature_perturbation=feature_perturbation, feature_names=feature_names, **deprecated_options)
        # change original model to the pipeline so it is used for predictions
        # self.model.original_model = self._original_model

def get_shap_values(estimator, X: pd.DataFrame, X_train: Optional[pd.DataFrame] = None, bytes: Optional[bool] = True):
    model_repr = str(estimator)
    if not 'Pipeline' in model_repr and ('XGB' in model_repr or 'Forest' in model_repr):
        explainer = shap.TreeExplainer(model=estimator)
    elif 'Pipeline' in model_repr and ('XGB' in model_repr or 'Forest' in model_repr):
        explainer = PipelineTree(model=estimator)
    else:
        replacement_data = X_train.astype(np.float64)
        # replacement_data = shap.kmeans(X=replacement_data, k=5)
        explainer = shap.KernelExplainer(model=estimator.predict, data=replacement_data)

    shap_per_class = explainer.shap_values(X.astype(np.float64))

    if not bytes:
        return shap_per_class, explainer.expected_value
    output = io.BytesIO()
    np.savez_compressed(output, shap_per_class, explainer.expected_value)
    shap_str = base64.b64encode(output.getvalue()).decode()

    return shap_str


def parse_shap_values(shap_str):
    npz_string = base64.b64decode(shap_str)
    input = io.BytesIO(npz_string)
    shap_values = np.load(input)
    return [val for val in shap_values.f.arr_0], [val for val in shap_values.f.arr_1]


