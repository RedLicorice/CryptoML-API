import pandas as pd
import numpy as np
from cryptoml.features.technical_indicators import get_ta_features, exponential_moving_average, simple_moving_average
from cryptoml.features.lagging import make_lagged
from cryptoml.features.spline import get_spline
from cryptoml.features.ohlcv_resample import ohlcv_resample
from cryptoml.util.convergence import convergence_between_series

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

    # Cherry-picked and engineered features from Blockchain data
    cm_picks = pd.DataFrame(index=ohlcv.index)
    if 'adractcnt' in coinmetrics.columns:
        cm_picks['adractcnt_pct'] = coinmetrics.adractcnt.pct_change()
    if 'txtfrvaladjntv' in coinmetrics.columns and 'isstotntv' in coinmetrics.columns and 'feetotntv' in coinmetrics.columns:
        # I want to represent miners earnings (fees + issued coins) vs amount transacted in that interval
        cm_picks['earned_vs_transacted'] = (coinmetrics.isstotntv + coinmetrics.feetotntv) / coinmetrics.txtfrvaladjntv
    if 'isstotntv' in coinmetrics.columns:
        # isstotntv is total number of coins mined in the time interval
        # splycur is total number of coins mined (all time)
        total_mined = coinmetrics.isstotntv.rolling(365, min_periods=7).sum()  # total mined in a year
        cm_picks['isstot1_isstot365_pct'] = (coinmetrics.isstotntv / total_mined).pct_change()
    if 'splycur' in coinmetrics.columns and 'isstotntv' in coinmetrics.columns:
        cm_picks['splycur_isstot1_pct'] = (coinmetrics.isstotntv / coinmetrics.splycur).pct_change()
    if 'hashrate' in coinmetrics.columns:
        cm_picks['hashrate_pct'] = coinmetrics.hashrate.pct_change()
    if 'roi30d' in coinmetrics.columns:
        cm_picks['roi30d'] = coinmetrics.roi30d
    if 'isstotntv' in coinmetrics.columns:
        cm_picks['isstotntv_pct'] = coinmetrics.isstotntv.pct_change()
    if 'feetotntv' in coinmetrics.columns:
        cm_picks['feetotntv_pct'] = coinmetrics.feetotntv.pct_change()
    if 'txtfrcount' in coinmetrics.columns:
        cm_picks['txtfrcount_pct'] = coinmetrics.txtfrcount.pct_change()
    if 'vtydayret30d' in coinmetrics.columns:
        cm_picks['vtydayret30d'] = coinmetrics.vtydayret30d
    if 'isscontpctann' in coinmetrics.columns:
        cm_picks['isscontpctann'] = coinmetrics.isscontpctann

    # Cherry-picked and engineered features from technical indicators
    ta = get_ta_features(ohlcv, TA_CONFIG)
    ta_picks = pd.DataFrame(index=ta.index)
    # REMA / RSMA are already used and well-estabilished in ATSA,
    # I'm taking the pct change since i want to encode the relative movement of the ema's not their positions
    # Drop other dimensions since they're correlated
    ta_picks['rema_8_15_pct'] = ta.rema_8_15.pct_change()
    ta_picks['rsma_8_15_pct'] = ta.rema_8_15.pct_change()

    # Stoch is a momentum indicator comparing a particular closing price of a security to a range of its prices
    # over a certain period of time.
    # The sensitivity of the oscillator to market movements is reducible by adjusting that time period or
    # by taking a moving average of the result.
    # It is used to generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.
    # IDEA => decrease sensitivity by 3-mean and divide by 100 to get fp values
    ta_picks['stoch_14_mean3_div100'] = ta.stoch_14.rolling(3).mean() / 100

    # Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows
    # the relationship between two moving averages of a security’s price.
    # The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
    #  A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line,
    #  which can function as a trigger for buy and sell signals.
    #  Traders may buy the security when the MACD crosses above its signal line and sell - or short - the security
    #  when the MACD crosses below the signal line.
    #  Moving Average Convergence Divergence (MACD) indicators can be interpreted in several ways,
    #  but the more common methods are crossovers, divergences, and rapid rises/falls.
    signal_line = exponential_moving_average(ta.macd_12_26, 9)
    ta_picks['macd_12_26_signal'] = signal_line
    ta_picks['macd_12_26_diff_signal'] = (ta.macd_12_26 - signal_line).pct_change()
    ta_picks['macd_12_26_pct'] = ta.macd_12_26.pct_change()

    # PPO is identical to the moving average convergence divergence (MACD) indicator,
    # except the PPO measures percentage difference between two EMAs, while the MACD measures absolute (dollar) difference.
    signal_line = exponential_moving_average(ta.ppo_12_26, 9)
    ta_picks['ppo_12_26_signal'] = signal_line
    ta_picks['ppo_12_26_diff_signal'] = (ta.ppo_12_26 - signal_line).pct_change()
    ta_picks['ppo_12_26_pct'] = ta.ppo_12_26.pct_change()

    # ADI Accumulation/distribution is a cumulative indicator that uses volume and price to assess whether
    # a stock is being accumulated or distributed.
    # The accumulation/distribution measure seeks to identify divergences between the stock price and volume flow.
    # This provides insight into how strong a trend is. If the price is rising but the indicator is falling
    # this indicates that buying or accumulation volume may not be enough to support
    # the price rise and a price decline could be forthcoming.
    # ==> IDEA: if we can fit a line to the price y1 = m1X+q1 and a line to ADI y2=m2X+q2 then we can identify
    #           divergences by simply looking at the sign of M.
    #           Another insight would be given by the slope (ie pct_change)
    ta_picks['adi_pct'] = ta.adi.pct_change()
    ta_picks['adi_close_convergence'] = convergence_between_series(ta.adi, ohlcv.close, 3)

    # RSI goes from 0 to 100, values <= 20 mean BUY, while values >= 80 mean SELL.
    # Dividing it by 100 to get a floating point feature, makes no sense to pct_change it
    ta_picks['rsi_14_div100'] = ta.rsi_14 / 100

    # The Money Flow Index (MFI) is a technical indicator that generates overbought or oversold
    #   signals using both prices and volume data. The oscillator moves between 0 and 100.
    # An MFI reading above 80 is considered overbought and an MFI reading below 20 is considered oversold,
    #   although levels of 90 and 10 are also used as thresholds.
    # A divergence between the indicator and price is noteworthy. For example, if the indicator is rising while
    #   the price is falling or flat, the price could start rising.
    ta_picks['mfi_14_div100'] = ta.mfi_14 / 100

    # The Chande momentum oscillator is a technical momentum indicator similar to other momentum indicators
    #   such as Wilder’s Relative Strength Index (Wilder’s RSI) and the Stochastic Oscillator.
    #   It measures momentum on both up and down days and does not smooth results, triggering more frequent
    #   oversold and overbought penetrations. The indicator oscillates between +100 and -100.
    # Many technical traders add a 10-period moving average to this oscillator to act as a signal line.
    #   The oscillator generates a bullish signal when it crosses above the moving average and a
    #   bearish signal when it drops below the moving average.
    ta_picks['cmo_14_div100'] = ta.cmo_14 / 100
    signal_line = simple_moving_average(ta.cmo_14, 10)
    ta_picks['cmo_14_signal'] = signal_line
    ta_picks['cmo_14_diff_signal'] = (ta.cmo_14 - signal_line) / 100

    # On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.
    # Eventually, volume drives the price upward. At that point, larger investors begin to sell, and smaller investors begin buying.
    # Despite being plotted on a price chart and measured numerically,
    # the actual individual quantitative value of OBV is not relevant.
    # The indicator itself is cumulative, while the time interval remains fixed by a dedicated starting point,
    # meaning the real number value of OBV arbitrarily depends on the start date.
    # Instead, traders and analysts look to the nature of OBV movements over time;
    # the slope of the OBV line carries all of the weight of analysis. => We want percent change
    ta_picks['obv_pct'] = ta.obv.pct_change()
    ta_picks['obv_mean3_pct'] = ta.obv.rolling(3).mean().pct_change()

    # Strong rallies in price should see the force index rise.
    # During pullbacks and sideways movements, the force index will often fall because the volume
    # and/or the size of the price moves gets smaller.
    # => Encoding the percent variation could be a good idea
    ta_picks['fi_13_pct'] = ta.fi_13.pct_change()
    ta_picks['fi_50_pct'] = ta.fi_50.pct_change()

    # The Aroon Oscillator is a trend-following indicator that uses aspects of the
    # Aroon Indicator (Aroon Up and Aroon Down) to gauge the strength of a current trend
    # and the likelihood that it will continue.
    # It moves between -100 and 100. A high oscillator value is an indication of an uptrend
    # while a low oscillator value is an indication of a downtrend.
    ta_picks['ao_14'] = ta.ao_14 / 100

    # The average true range (ATR) is a technical analysis indicator that measures market volatility
    #   by decomposing the entire range of an asset price for that period.
    # ATRP is pct_change of volatility
    ta_picks['atrp_14'] = ta.atrp_14

    # Percentage Volume Oscillator (PVO) is momentum volume oscillator used in technical analysis
    #   to evaluate and measure volume surges and to compare trading volume to the average longer-term volume.
    # PVO does not analyze price and it is based solely on volume.
    #  It compares fast and slow volume moving averages by showing how short-term volume differs from
    #  the average volume over longer-term.
    #  Since it does not care a trend's factor in its calculation (only volume data are used)
    #  this technical indicator cannot be used alone to predict changes in a trend.
    ta_picks['pvo_12_26'] = ta.pvo_12_26

    # Lagged percent variation of OHLCV
    ohlcv_pct = ohlcv[['open', 'high', 'low', 'close', 'volume']].pct_change()
    ohlcv_pct.columns = ['{}_pct'.format(c) for c in ohlcv_pct.columns]
    lagged_ohlcv_pct = pd.concat(
        [ohlcv_pct] + [make_lagged(ohlcv_pct, i) for i in range(1, W + 1)],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )

    # Use SPLINES to extract price information
    ohlc_splines = pd.DataFrame(index=ohlcv.index)
    # First derivative indicates slope
    ohlc_splines['open_spl_d1'] = get_spline(ohlcv.open, 1)
    ohlc_splines['high_spl_d1'] = get_spline(ohlcv.high, 1)
    ohlc_splines['low_spl_d1'] = get_spline(ohlcv.low, 1)
    ohlc_splines['close_spl_d1'] = get_spline(ohlcv.close, 1)
    # Second derivative indicates convexity
    ohlc_splines['open_spl_d2'] = get_spline(ohlcv.open, 2)
    ohlc_splines['high_spl_d2'] = get_spline(ohlcv.high, 2)
    ohlc_splines['low_spl_d2'] = get_spline(ohlcv.low, 2)
    ohlc_splines['close_spl_d2'] = get_spline(ohlcv.close, 2)

    # OHLC Stats
    ohlcv_stats = pd.DataFrame(index=ohlcv.index)
    ohlcv_stats['close_open_pct'] = (ohlcv.close - ohlcv.open).pct_change()  # Change in body of the candle (> 0 if candle is green)
    ohlcv_stats['high_close_dist_pct'] = (ohlcv.high - ohlcv.close).pct_change()  # Change in wick size of the candle, shorter wick should be bullish
    ohlcv_stats['low_close_dist_pct'] = (ohlcv.close - ohlcv.low).pct_change()  # Change in shadow size of the candle, this increasing would indicate support (maybe a bounce)
    ohlcv_stats['high_low_dist_pct'] = (ohlcv.high - ohlcv.low).pct_change()  # Change in total candle size, smaller candles stands for low volatility
    ohlcv_stats['close_volatility_7d'] = ohlcv.close.pct_change().rolling(7).std(ddof=0)
    ohlcv_stats['close_volatility_30d'] = ohlcv.close.pct_change().rolling(30).std(ddof=0)

    for d in [3, 7, 30]:
        ohlcv_d = ohlcv_resample(ohlcv=ohlcv, period=d, interval='D')
        ohlcv_stats['close_open_pct_d{}'.format(d)] = (ohlcv_d[d].close - ohlcv_d[d].open).pct_change()
        ohlcv_stats['high_close_dist_pct_d{}'.format(d)] = (ohlcv_d[d].high - ohlcv_d[d].close).pct_change()
        ohlcv_stats['low_close_dist_pct_d{}'.format(d)] = (ohlcv_d[d].close - ohlcv_d[d].low).pct_change()
        ohlcv_stats['high_low_dist_pct_d{}'.format(d)] = (ohlcv_d[d].high - ohlcv_d[d].low).pct_change()

    return pd.concat(
        [ohlc_splines, ohlcv_stats, lagged_ohlcv_pct, cm_picks, ta_picks],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )


