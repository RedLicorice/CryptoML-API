from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.trading_service import TradingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.series_encoding import onehot_target
from cryptoml_core.util.timestamp import get_timestamps
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.dates as mpl_dates
import mplfinance as mpf
import typer

def main(pipeline: str, dataset: str, symbol: str, window: int):
    ds = DatasetService()
    ms = ModelService()
    ts = TradingService()
    ohlcv_ds = ds.get_dataset('ohlcv', symbol=symbol)
    asset = ts.get_asset(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol, window=window, create=False)
    if not asset:
        print(f"Asset {pipeline}.{dataset}.class for {symbol} on window {window} not found!")
        return
    test = ms.get_test(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol, window=window)
    if not test:
        print(f"Test {pipeline}.{dataset}.class for {symbol} on window {window} not found!")
    # ohlcv = ohlcv.loc[test.test_interval.begin:test.test_interval.end]
    ohlcv = ds.get_dataset_features(ohlcv_ds, begin=test.test_interval.begin, end=test.test_interval.end)
    test_results = ModelService.parse_test_results(test).iloc[:-1]
    enc_label = onehot_target(test_results.label, labels=["is_sell", "is_hold", "is_buy"], fill=False)
    enc_pred = onehot_target(test_results.predicted, labels=["is_sell", "is_hold", "is_buy"], fill=False)

    # Mask predictions with low value minus a certain amount
    signals_level_diff = ohlcv.low * 10 / 100
    signals_level = ohlcv.low - signals_level_diff
    #signals_level = ohlcv.low
    enc_pred.is_sell.mask(enc_pred.is_sell > 0, other=signals_level, inplace=True)
    enc_pred.is_hold.mask(enc_pred.is_hold > 0, other=signals_level, inplace=True)
    enc_pred.is_buy.mask(enc_pred.is_buy > 0, other=signals_level, inplace=True)

    # Get unique years in index to split plots in smaller scale
    unique_years = ohlcv.index.year.unique()
    for year in unique_years:
        year_pred = enc_pred[enc_pred.index.year == year]
        year_ohlcv = ohlcv[ohlcv.index.year == year]

        # Set up xticks
        daysToIndex = {ts.to_pydatetime(): i for i, ts in enumerate(year_ohlcv.index)}
        days = [i for i in daysToIndex.values()]
        labels = [ts.to_pydatetime().strftime("%Y-%m-%d") for ts in year_ohlcv.index]


        # Setup matplotfinance styles and figure
        s = mpf.make_mpf_style(base_mpf_style='binance')  # , rc={'font.size': 6}
        fig = mpf.figure(figsize=(16, 8), style=s)  # pass in the self defined style to the whole canvas
        fig.suptitle(f"{ohlcv_ds.symbol}, {year}, 1D")

        ax = fig.add_subplot(3, 1, (1, 2))  # main candle stick chart subplot
        av = fig.add_subplot(3, 1, 3, sharex=ax)  # volume candles subplot

        # Setup horizontal grids
        ax.grid(axis='x', color='0.5', linestyle='--')
        av.grid(axis='x', color='0.5', linestyle='--')

        # for a in [ax, av]:
        #     a.set_xticks(ticks=days)
        #     a.set_xticklabels(labels=labels)
        #     a.tick_params(axis='x', labelrotation=90)

        apds = [
            #     mpf.make_addplot(tcdf)
            # Predictions
            mpf.make_addplot(year_ohlcv.close, ax=ax, type='line', color=(0.5, 0.5, 0.5, 0.05)),
            mpf.make_addplot(year_pred.is_sell, ax=ax, type='scatter', marker='v', color='red'),
            mpf.make_addplot(year_pred.is_hold, ax=ax, type='scatter', marker='_', color='silver'),
            mpf.make_addplot(year_pred.is_buy, ax=ax, type='scatter', marker='^', color='lime'),
        ]

        mpf.plot(
            year_ohlcv,
            type='candle',
            style=s,
            #ylabel='Price ($)',
            ax=ax,
            volume=av,
            #ylabel_lower='Volume',
            show_nontrading=True,
            addplot=apds,
            returnfig=True
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        print("Done")


if __name__ == '__main__':
    typer.run(main)
