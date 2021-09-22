from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.trading_service import TradingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.series_encoding import onehot_target
from cryptoml_core.util.timestamp import get_timestamps
from cryptoml.features.technical_indicators import upper_bollinger_band, middle_bollinger_band, lower_bollinger_band

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import typer
import os


SYMBOLS = [
    "ADAUSD", "BCHUSD", "BNBUSD",
    "BTCUSD", "BTGUSD", "DASHUSD",
    "DOGEUSD", "EOSUSD", "ETCUSD",
    "ETHUSD", "LINKUSD", "LTCUSD",
    "NEOUSD", "QTUMUSD", "TRXUSD",
    "VENUSD", "WAVESUSD", "XEMUSD",
    "XMRUSD", "XRPUSD", "ZECUSD",
    "ZRXUSD"
]

WINDOWS = [ 90, 180, 240 ]

PIPELINES = [
    # 'adaboost_decisiontree',
    'bagging_decisiontree',
    # 'bagging_linear_svc',
    'bagging_poly_svc',
    # 'bagging_rbf_svc',
    # 'debug_xgboost',
    'plain_knn',
    # 'plain_linear_svc',
    'plain_mlp',
    #'plain_mlp_big',
    'plain_mnb',
    'plain_poly_svc',
    'plain_randomforest',
    # 'plain_rbf_svc',
    'plain_xgboost',
    'smote_knn',
    'smote_mlp',
    'smote_poly_svc'
]

def make_plot(ohlcv: pd.DataFrame, orders: pd.DataFrame, equity: pd.DataFrame, pred: pd.Series, **kwargs):
    fig = make_subplots(rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(
                            kwargs.get("signals_title", "Signals"),
                            kwargs.get("longs_title", "Longs"),
                            kwargs.get("shorts_title", "Shorts"),
                            kwargs.get("equity_title", "Equity")
                        ),
                        row_heights=[0.25, 0.25, 0.25, 0.25],
                        specs=[
                            [{"secondary_y": True}],
                            [{"secondary_y": False}],
                            [{"secondary_y": False}],
                            [{"secondary_y": False}]
                        ]
                        )
    # Candlestick plot w/ volume overlay for signals
    go_cdl = go.Candlestick(x=ohlcv.index,
                            open=ohlcv.open,
                            high=ohlcv.high,
                            low=ohlcv.low,
                            close=ohlcv.close,
                            name='OHLC',
                            increasing_line_color='gray',
                            decreasing_line_color='black',
                            showlegend=False
                            )
    go_vol = go.Bar(x=ohlcv.index,
                    y=ohlcv.volume,
                    showlegend=False,
                    name='Volume',
                    marker_color='darkblue',
                    opacity=0.3
                    )

    fig.add_trace(go_cdl, secondary_y=True, row=1, col=1)  # Signals OHLCV Plot
    fig.add_trace(go_vol, secondary_y=False, row=1, col=1)  # Signals Volume Plot

    # Plot Signals on first OHLCV Plot
    fig.add_trace(go.Scatter(
        x=pred.index,
        y=pred.is_sell,
        mode='markers',
        marker={
            'size': 6,
            'color': 'red',
            'symbol': 'arrow-down'
        },
        name='Sell'
    ), secondary_y=True, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred.index,
        y=pred.is_hold,
        mode='markers',
        marker={
            'size': 6,
            'color': 'purple',
            'symbol': 'x'
        },
        name='Hold'
    ), secondary_y=True, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred.index,
        y=pred.is_buy,
        mode='markers',
        marker={
            'size': 6,
            'color': 'green',
            'symbol': 'arrow-up'
        },
        name='Buy'
    ), secondary_y=True, row=1, col=1)

    # Candlestick plot for long orders
    fig.add_trace(go_cdl, row=2, col=1)  # Trades OHLCV Plot
    fig.add_trace(go.Scatter(
        x=orders.index,
        y=orders.price.mask(orders.type != 'OPEN_LONG'),
        mode='markers',
        marker={
            'size': 6,
            'color': orders.position_id,
            'colorscale': px.colors.qualitative.Light24,
            'symbol': 'circle-open'
        },
        name='Open Long'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=orders.index,
        y=orders.price.mask(orders.type != 'CLOSE_LONG'),
        mode='markers',
        marker={
            'size': 4,
            'color': orders.position_id,
            'colorscale': px.colors.qualitative.Light24,
            'symbol': 'circle'
        },
        name='Close Long'
    ), row=2, col=1)
    # Candlestick plot for short orders
    fig.add_trace(go_cdl, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=orders.index,
        y=orders.price.mask(orders.type != 'OPEN_SHORT'),
        mode='markers',
        marker={
            'size': 8,
            'color': orders.position_id,
            'colorscale': px.colors.qualitative.Light24,
            'symbol': 'circle-open'
        },
        name='Open Short'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=orders.index,
        y=orders.price.mask(orders.type != 'CLOSE_SHORT'),
        mode='markers',
        marker={
            'size': 4,
            'color': orders.position_id,
            'colorscale': px.colors.qualitative.Light24,
            'symbol': 'circle'
        },
        name='Close Short'
    ), row=3, col=1)

    # Equity plot
    go_eq = go.Scatter(x=equity.index, y=equity.equity, mode='lines', name='Equity')
    fig.add_trace(go_eq, row=4, col=1)
    baselines = kwargs.get('baselines')
    for name, b in baselines:
        go_eq = go.Scatter(x=b.index, y=b.equity, mode='lines', name=name)
        fig.add_trace(go_eq, row=4, col=1)

    if kwargs.get('annotate'):
        for index, order in orders[orders.type == 'CLOSE_LONG'].iterrows():
            fig.add_annotation(
                x=index,
                y=order.price,
                text=f"{get_initials(order.detail)} {order.change}%",
                showarrow=True,
                arrowhead=1,
                row=2,
                col=1
            )
            fig.add_annotation(
                x=index,
                y=equity.loc[index].equity,
                text=f"{get_initials(order.detail)} {order.change}%",
                showarrow=True,
                arrowhead=1,
                row=4,
                col=1
            )
        for index, order in orders[orders.type == 'CLOSE_SHORT'].iterrows():
            fig.add_annotation(
                x=index,
                y=order.price,
                text=f"{get_initials(order.detail)} {order.change}%",
                showarrow=True,
                arrowhead=1,
                row=3,
                col=1
            )
            fig.add_annotation(
                x=index,
                y=equity.loc[index].equity,
                text=f"{get_initials(order.detail)} {order.change}%",
                showarrow=True,
                arrowhead=1,
                row=4,
                col=1
            )
    # _period = 21
    # if kwargs.get("bollinger") and ohlcv.shape[0] > _period:
    #     go_boll_hi = go.Scatter(
    #         x=ohlcv.index,
    #         y=upper_bollinger_band(ohlcv.close, _period),
    #         mode='lines',
    #         name='BB+',
    #         line={
    #             'width': 1,
    #             'color': 'darkgreen',
    #         },
    #         opacity=0.5,
    #         showlegend=False
    #     )
    #     go_boll_mid = go.Scatter(
    #         x=ohlcv.index,
    #         y=middle_bollinger_band(ohlcv.close, _period),
    #         mode='lines',
    #         name='BB',
    #         line={
    #             'width': 1,
    #             'color': 'darkslategrey',
    #         },
    #         opacity=0.5,
    #         showlegend=False
    #     )
    #     go_boll_low = go.Scatter(
    #         x=ohlcv.index,
    #         y=lower_bollinger_band(ohlcv.close, _period),
    #         mode='lines',
    #         name='BB-',
    #         line={
    #             'width': 1,
    #             'color': 'darkred',
    #         },
    #         opacity=0.5,
    #         showlegend=False
    #     )
    #     fig.add_trace(go_boll_hi, secondary_y=False, row=2, col=1)
    #     fig.add_trace(go_boll_mid, secondary_y=False, row=2, col=1)
    #     fig.add_trace(go_boll_low, secondary_y=False, row=2, col=1)
    #     fig.add_trace(go_boll_hi, secondary_y=False, row=3, col=1)
    #     fig.add_trace(go_boll_mid, secondary_y=False, row=3, col=1)
    #     fig.add_trace(go_boll_low, secondary_y=False, row=3, col=1)



    # Set up xticks
    fig.update_xaxes(**{
        # 'dtick': 1,
        # 'tickmode': 'linear',
        'tickformat': '%d-%m-%Y',
        'rangeslider_visible': False  # Do not show OHLC's range slider
    })
    fig.update_yaxes(**{
        'dtick': 1,
        'tickmode': 'auto'
    })

    # fig.show()
    imgPath = kwargs.get("img_path")
    imgName = kwargs.get("img_name")
    if imgPath and imgName:
        os.makedirs(imgPath, exist_ok=True)
        fig.write_image(imgPath + imgName, width=1024, height=900, scale=1)
    if kwargs.get("show"):
        fig.show()


def get_initials(fullname):
    if not fullname:
        return ""
    xs = (fullname)
    name_list = xs.split()

    initials = ""

    for name in name_list:  # go through each name
        initials += name[0].upper()  # append the initial

    return initials


def main(dataset: str):
    ds = DatasetService()
    ms = ModelService()
    ts = TradingService()
    logs = []
    for pipeline in PIPELINES:
        for symbol in SYMBOLS:
            for window in WINDOWS:
                print(f"PIPELINE: {pipeline} SYMBOL: {symbol} WINDOW: {window}")
                ohlcv_ds = ds.get_dataset('ohlcv', symbol=symbol)
                test = ms.get_test(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol, window=window)
                if not test:
                    print(f"Test {pipeline}.{dataset}.class for {symbol} on window {window} not found!")
                    logs.append(
                        f"MISSING_TEST {pipeline} {dataset} {symbol} class {window} --features importances_shap --parameters gridsearch\n")
                    continue
                asset = ts.get_asset(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol, window=window, create=False)
                if not asset:
                    print(f"Asset {pipeline}.{dataset}.class for {symbol} on window {window} not found!")
                    logs.append(f"MISSING_ASSET {pipeline} {dataset} {symbol} {window}\n")
                    continue

                equity = TradingService.parse_equity_df(asset=asset)
                buy_and_hold = TradingService.parse_baseline_df(asset=asset, name='buy_and_hold')
                orders = TradingService.parse_orders_df(asset=asset)

                # Map order position_id to numbers so we don't get a mess in the graph
                position_uids = set(orders.position_id.values)
                for i, uid in enumerate(position_uids):
                    orders.position_id.replace(to_replace=uid, value=i, inplace=True)

                ohlcv = ds.get_dataset_features(ohlcv_ds, begin=test.test_interval.begin, end=test.test_interval.end)
                test_results = ModelService.parse_test_results(test).iloc[:-1]
                # Mask predictions with low value minus a certain amount
                signals_level_diff = ohlcv.low * 10 / 100
                signals_level = ohlcv.low - signals_level_diff
                enc_pred = onehot_target(test_results.predicted, labels=["is_sell", "is_hold", "is_buy"], fill=False)
                #  In case of classifier bias (due to input bias) some classes are ignored.
                # In such cases, enc_pred won't contain the ignored classes.
                # Add them back by nan-filling (never selected)
                if hasattr(enc_pred, 'is_sell'):
                    use_idx = enc_pred.is_sell > 0
                    enc_pred.is_sell.mask(use_idx, other=signals_level.loc[enc_pred.index], inplace=True)
                else:
                    enc_pred['is_sell'] = pd.Series(np.nan, index=enc_pred.index)
                if hasattr(enc_pred, 'is_hold'):
                    enc_pred.is_hold.mask(enc_pred.is_hold > 0, other=signals_level.loc[enc_pred.index], inplace=True)
                else:
                    enc_pred['is_hold'] = pd.Series(np.nan, index=enc_pred.index)
                if hasattr(enc_pred, 'is_buy'):
                    enc_pred.is_buy.mask(enc_pred.is_buy > 0, other=signals_level.loc[enc_pred.index], inplace=True)
                else:
                    enc_pred['is_buy'] = pd.Series(np.nan, index=enc_pred.index)

                # Get unique years in index to split plots in smaller scale
                unique_years = ohlcv.index.year.unique()
                for year in unique_years:
                    year_ohlcv = ohlcv[ohlcv.index.year == year]
                    year_pred = enc_pred[enc_pred.index.year == year]
                    year_equity = equity[equity.index.year == year]
                    year_buy_and_hodl = buy_and_hold[buy_and_hold.index.year == year]
                    year_orders = orders[orders.index.year == year]

                    unique_quarters = year_ohlcv.index.quarter.unique()
                    for quarter in unique_quarters:
                        q_ohlcv = year_ohlcv[year_ohlcv.index.quarter == quarter]
                        q_pred = year_pred[year_pred.index.quarter == quarter]
                        q_equity = year_equity[year_equity.index.quarter == quarter]
                        q_orders = year_orders[year_orders.index.quarter == quarter]
                        q_buy_and_hodl = year_buy_and_hodl[year_buy_and_hodl.index.quarter == quarter]
                        #f"{ohlcv_ds.symbol}, {year} - Q{quarter}, 1D", 'Trades', 'Equity'
                        img_path = f"images/backtests-final/{pipeline}-{dataset}-class-W{window}/{symbol}/"
                        img_name = f"trades-{year}-Q{quarter}.png"
                        if os.path.exists(f"{img_path}/{img_name}"):
                            print(f"[SKIP] File exists {img_path}/{img_name}")
                            continue
                        make_plot(
                            ohlcv=q_ohlcv,
                            orders=q_orders,
                            equity=q_equity,
                            baselines=[
                                ('Buy and Hold', q_buy_and_hodl)
                            ],
                            pred=q_pred,
                            signals_title=f"{ohlcv_ds.symbol}, {pipeline}, W={window}D, {year} - Q{quarter}, 1D",
                            img_path=img_path,
                            img_name=img_name,
                            bollinger=True
                        )
                        print(f"{year}-Q{quarter} saved to {img_path}{img_name}")
    with open(f"trading_plotly.{dataset}.log", "w") as f:
        f.writelines(logs)
    print("Logs saved")


if __name__ == '__main__':
    typer.run(main)
