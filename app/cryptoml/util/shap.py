import shap
import pandas as pd
import numpy as np
import io
import base64


def get_shap_values(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model=model)
    shap_per_class = explainer.shap_values(X.astype(np.float64))

    output = io.BytesIO()
    np.savez_compressed(output, shap_per_class, explainer.expected_value)
    shap_str = base64.b64encode(output.getvalue()).decode()

    return shap_str


def parse_shap_values(shap_str):
    npz_string = base64.b64decode(shap_str)
    input = io.BytesIO(npz_string)
    shap_values = np.load(input)
    return [val for val in shap_values.f.arr_0], [val for val in shap_values.f.arr_1]


