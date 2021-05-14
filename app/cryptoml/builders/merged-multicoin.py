import pandas as pd
import numpy as np
from cryptoml.features.technical_indicators import get_ta_features, exponential_moving_average, simple_moving_average
from cryptoml.features.talib import get_talib_patterns
from cryptoml.features.spline import get_spline
from cryptoml.features.lagging import make_lagged
from cryptoml.features.decompose import get_residual
from cryptoml.features.ohlcv import ohlcv_resample
from cryptoml.util.convergence import convergence_between_series


def build(source: pd.DataFrame, *targets, **kwargs):
   result = pd.concat([source] +  targets, axis='columns')
   return result

