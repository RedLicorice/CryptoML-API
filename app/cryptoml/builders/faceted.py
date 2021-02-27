import pandas as pd
import numpy as np
from cryptoml.features.technical_indicators import get_ta_features
from cryptoml.features.lagging import make_lagged
from cryptoml.features.decompose import get_residual

TA_CONFIG = {
    'rsma': [(5, 20), (8, 15), (20, 50)],
    'rema': [(5, 20), (8, 15), (20, 50)],
    'macd': [(12, 26)],
    'ao': [14],
    'adx': [14],
    'wd': [14],
    'ppo': [(12, 26)],
    'rsi': [14],
    'mfi': [14],
    'tsi': None,
    'stoch': [14],
    'cmo': [14],
    'atrp': [14],
    'pvo': [(12, 26)],
    'fi': [13, 50],
    'adi': None,
    'obv': None
}


def build(ohlcv: pd.DataFrame, coinmetrics: pd.DataFrame, **kwargs):
    W = kwargs.get('W', 10)
    ta = get_ta_features(ohlcv, TA_CONFIG)

    ohlc_pct = ohlcv[['open', 'high', 'low', 'close']].pct_change()
    ohlc_pct.columns = ["{}_pct".format(c) for c in ohlc_pct.columns]
    history_facet = pd.concat(
        [ohlc_pct] + [make_lagged(ohlc_pct, i) for i in range(1, W + 1)],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )

    # Price trend facet (REMA/RSMA, MACD, AO, ADX, WD+ - WD-)
    trend_facet = ta[[
        "rsma_5_20", "rsma_8_15", "rsma_20_50",
        "rema_5_20", "rema_8_15", "rema_20_50",
        "macd_12_26", "ao_14", "adx_14", "wd_14"
    ]]
    # Volatility facet (CMO, ATRp)
    volatility_facet = ta[["cmo_14", "atrp_14"]]
    # Volume facet (Volume pct, PVO, ADI, OBV)
    volume_pct = ohlcv.volume.pct_change().replace([np.inf, -np.inf], 0)
    volume_facet = pd.concat(
        [volume_pct, ta[["pvo_12_26", "adi", "obv"]]],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )
    # On-chain facet
    cm_1 = coinmetrics.reindex(columns=[
        'adractcnt', 'txtfrvaladjntv', 'isstotntv',
        'feetotntv', 'splycur', 'hashrate',
        'difficulty', 'txtfrcount']) \
        .pct_change()
    cm_2 = coinmetrics.reindex(columns=['isscontpctann'])
    chain_facet = pd.concat([cm_1, cm_2], axis='columns', verify_integrity=True, sort=True, join='inner')

    # Drop columns whose values are all nan or inf from each facet
    with pd.option_context('mode.use_inf_as_na', True):  # Set option temporarily
        history_facet = history_facet.dropna(axis='columns', how='all')
        trend_facet = trend_facet.dropna(axis='columns', how='all')
        volatility_facet = volatility_facet.dropna(axis='columns', how='all')
        volume_facet = volume_facet.dropna(axis='columns', how='all')
        chain_facet = chain_facet.dropna(axis='columns', how='all')

    # feature_groups = {
    #     'price_history': [c for c in history_facet.columns],
    #     'trend': [c for c in trend_facet.columns],
    #     'volatility': [c for c in volatility_facet.columns],
    #     'volume': [c for c in volume_facet.columns],
    #     'chain': [c for c in chain_facet.columns],
    # }

    return pd.concat(
        [history_facet, trend_facet, volatility_facet, volume_facet, chain_facet],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )
