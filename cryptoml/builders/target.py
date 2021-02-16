import pandas as pd
from cryptoml.features.targets import target_price, target_pct, target_class, target_binary, target_binned_class


def build(ohlcv: pd.DataFrame, **kwargs):
    result = pd.DataFrame(index=ohlcv.index)
    result['price'] = target_price(ohlcv.close)
    result['pct'] = target_pct(ohlcv.close)
    result['class'] = target_class(ohlcv.close)
    result['binary'] = target_binary(ohlcv.close)
    result['bin_class'] = target_binned_class(ohlcv.close, n_bins=3)
    result['bin_binary'] = target_binned_class(ohlcv.close, n_bins=2)
    return result