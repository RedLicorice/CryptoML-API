import pandas as pd
from cryptoml.features.lagging import make_lagged
from cryptoml.features.technical_indicators import get_ta_features

TA_CONFIG = {
	'rsma' : [(5,20), (8,15), (20,50)],
	'rema' : [(5,20), (8,15), (20,50)],
	'macd' : [(12,26)],
	'ao' : [14],
	'adx' : [14],
	'wd' : [14],
	'ppo' : [(12,26)],
	'rsi':[14],
	'mfi':[14],
	'tsi':None,
	'stoch':[14],
	'cmo':[14],
	'atrp':[14],
	'pvo':[(12,26)],
	'fi':[13,50],
	'adi':None,
	'obv':None
}

def build(ohlcv: pd.DataFrame, **kwargs):
    W = kwargs.get('W', 10)
    ohlc = ohlcv[['open', 'high', 'low', 'close']]
    lagged_ohlc = pd.concat(
        [ohlc] + [make_lagged(ohlc, i) for i in range(1, W + 1)],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )
    ta = get_ta_features(ohlcv, TA_CONFIG)
    return pd.concat([lagged_ohlc, ta], axis='columns', verify_integrity=True, sort=True, join='inner')