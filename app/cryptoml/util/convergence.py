import pandas as pd
import numpy as np
from scipy.stats import linregress


def convergence_between_series(s1: pd.Series, s2: pd.Series, W):
    # Fit a line on values in the window, then
    # return the angular coefficient's sign: 1 if positive, 0 otherwise
    def get_alpha(values: np.ndarray):
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(0, values.size), values)
        #alpha = np.arctan(slope) / (np.pi / 2)
        return 1 if slope > 0 else 0
    # Map both series to their direction (0 decreasing, 1 increasing)
    s1_ = s1.rolling(W).apply(get_alpha, raw=True)
    s2_ = s2.rolling(W).apply(get_alpha, raw=True)
    # Result should be true if both values have the same direction (either both 0's or both 1's)
    #  XOR has the inverse of the truth table we need, so we just negate it.
    result = np.logical_not(np.logical_xor(s1_, s2_)).astype(int)
    return result