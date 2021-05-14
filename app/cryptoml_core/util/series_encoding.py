import numpy as np
import pandas as pd
from typing import List


# One-hot encode a pandas.series to a dataframe using specified labels
def onehot_target(series: pd.Series, labels: List[str], fill=True):
    unique = np.unique(series)
    result = pd.DataFrame(index=series.index)
    for c in unique:
        label = labels[c]
        # other_values = [x for x in unique if x != c]
        # result[label] = series.replace(to_replace=other_values, value=np.nan)
        # result[label] = result[label].replace(to_replace=c, value=1)
        result[label] = series.mask(series != c).replace(to_replace=c, value=1)
    if fill:
        result = result.fillna(value=0)
    return result