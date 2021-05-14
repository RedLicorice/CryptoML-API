from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.trading_service import TradingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.series_encoding import onehot_target
from cryptoml_core.util.timestamp import get_timestamps
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    equity = ts.parse_equity_df(asset=asset)

    ohlcv = ds.get_dataset_features(ohlcv_ds, begin=test.test_interval.begin, end=test.test_interval.end)
    test_results = ModelService.parse_test_results(test).iloc[:-1]
    # Mask predictions with low value minus a certain amount
    signals_level_diff = ohlcv.low * 10 / 100
    signals_level = ohlcv.low - signals_level_diff
    enc_pred = onehot_target(test_results.predicted, labels=["is_sell", "is_hold", "is_buy"], fill=False)
    enc_pred.is_sell.mask(enc_pred.is_sell > 0, other=signals_level, inplace=True)
    enc_pred.is_hold.mask(enc_pred.is_hold > 0, other=signals_level, inplace=True)
    enc_pred.is_buy.mask(enc_pred.is_buy > 0, other=signals_level, inplace=True)

    # Get unique years in index to split plots in smaller scale
    unique_years = ohlcv.index.year.unique()
    for year in unique_years:
        year_ohlcv = ohlcv[ohlcv.index.year == year]
        year_pred = enc_pred[enc_pred.index.year == year]
        year_equity = equity[equity.index.year == year]

        unique_quarters = year_ohlcv.index.quarter.unique()
        for quarter in unique_quarters:
            q_ohlcv = year_ohlcv[year_ohlcv.index.quarter == quarter]
            q_pred = year_pred[year_pred.index.quarter == quarter]
            q_equity = year_equity[year_equity.index.quarter == quarter]
            fig = make_subplots(rows=3, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.03,
                                subplot_titles=(f"{ohlcv_ds.symbol}, {year} - Q{quarter}, 1D", 'Volume', 'Equity'),
                                row_width=[0.3, 0.2, 0.5]
                                )
            # Candlestick plot
            go_cdl = go.Candlestick(x=q_ohlcv.index,
                                    open=q_ohlcv.open,
                                    high=q_ohlcv.high,
                                    low=q_ohlcv.low,
                                    close=q_ohlcv.close,
                                    name='OHLC'
                                    )
            fig.add_trace(go_cdl, row=1, col=1)
            # Add signals to candlestick plot
            fig.add_trace(go.Scatter(
                x=q_pred.index,
                y=q_pred.is_sell,
                mode='markers',
                marker={
                    'size': 8,
                    'color': 'red',
                    'symbol': 'arrow-down'
                },
                name='Sell'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=q_pred.index,
                y=q_pred.is_hold,
                mode='markers',
                marker={
                    'size': 8,
                    'color': 'gray',
                    'symbol': 'x'
                },
                name='Hold'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=q_pred.index,
                y=q_pred.is_buy,
                mode='markers',
                marker={
                    'size': 8,
                    'color': 'green',
                    'symbol': 'arrow-up'
                },
                name='Buy'
            ), row=1, col=1)

            # Volume plot
            fig.add_trace(go.Bar(x=q_ohlcv.index, y=q_ohlcv.volume, showlegend=False, name='Volume'), row=2, col=1)

            # Equity plot
            fig.add_trace(go.Scatter(x=q_equity.index, y=q_equity.equity, mode='lines', name='Equity'), row=3, col=1)

            # Set up xticks
            fig.update_layout(
                xaxis={
                    'dtick': 1,
                    'tickmode': 'linear',
                    'tickformat': '%d-%m-%Y',
                    'rangeslider_visible': False  # Do not show OHLC's range slider
                },
                yaxis={
                    'dtick': 1,
                    'tickmode': 'auto'
                }
            )

            fig.show()
            print(f"{year}-Q{quarter} done")


if __name__ == '__main__':
    typer.run(main)
