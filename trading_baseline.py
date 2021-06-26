import typer

from cryptoml_core.exceptions import MessageException
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.trading_service import TradingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.timestamp import to_timestamp
from cryptoml.features.technical_indicators import percent_b
from cryptoml.features.technical_indicators import volatility
from cryptoml.util.discretization import to_discrete_double
from sklearn.metrics import precision_score

import numpy as np
import pandas as pd

SELL, HOLD, BUY = range(3)
TARGETS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}

def main(pipeline: str, dataset: str, symbol: str, window: int):
    ds = DatasetService()
    ms = ModelService()
    ts = TradingService()
    ohlcv_ds = ds.get_dataset('ohlcv', symbol=symbol)
    ohlcv = ds.get_dataset_features(ohlcv_ds)  # [ohlcv_ds.valid_index_min:ohlcv_ds.valid_index_max]

    # boll = pd.Series(percent_b(ohlcv.close, 21), index=ohlcv.index)
    boll = pd.Series(
        to_discrete_double(percent_b(ohlcv.close, 21), 20, 80),
        index=ohlcv.index
    ).replace(to_replace=-1, value=np.nan)

    #model = ms.get_model(pipeline, dataset, 'class', symbol)
    _test = ms.get_test(pipeline, dataset, 'class', symbol, window)
    for test in [_test]:  # I originally traded all the tests in the model. ToDo: Refactor this.
        # Re-convert classification results from test to a DataFrame
        ohlcv_results = ohlcv[test.test_interval.begin:test.test_interval.end]
        results = ModelService.parse_test_results(test)

        #results.index = ohlcv_results.index
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        # results.index = pd.to_datetime(results.index)

        asset = ts.get_asset(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol,
                             window=test.window['days'])
        # Now use classification results to trade!
        day_count = results.shape[0]
        cur_day = 0
        print(
            "%B_Precision = {}",
            precision_score(results.label, boll.loc[results.index], average='macro', zero_division=0)
        )
        # Amount to buy in coins for buy and hold: $10k divided by first price in test set
        bh_price = ohlcv.close.loc[test.test_interval.begin]
        bh_amount = 10000 / bh_price

        for index, pred in results.iterrows():
            cur_day += 1
            # Get simulation day by converting Pandas' Timestamp to our format
            simulation_day = to_timestamp(index.to_pydatetime())
            # Results dataframe interprets values as float, while they are actually int
            predicted, label = int(pred.predicted), int(pred.label)

            # Grab ohlcv values for current day
            try:
                values = ohlcv.loc[index]
            except KeyError:
                print(f"Day: {index} not in OHLCV index!")
                continue
            try:
                boll_sig = boll.loc[index] if boll.loc[index] != np.nan else None
            except KeyError:
                boll_sig = None
                print(f"Day: {index} not in BOLL index!")
                pass
            _index = ohlcv.index.get_loc(index)
            change = TradingService.get_percent_change(values.close, values.open)

            print(f"Day {cur_day}/{day_count} [{index}] "
                  f"[O {values.open} H {values.high} L {values.low} C {values.close}] "
                  f"PCT={change}% "
                  f"LABEL={TARGETS[label]} BPRED={TARGETS[boll_sig]} PRED={TARGETS[predicted]}"
                  )
            open_positions = ts.get_open_positions(asset=asset, day=simulation_day)
            for p in open_positions:
                p_age = TradingService.get_position_age(position=p, day=simulation_day)
                try:
                    if p.type == 'MARGIN_LONG':
                        if TradingService.check_stop_loss(p, values.low):
                            ts.close_long(asset=asset, day=simulation_day, close_price=p.stop_loss, position=p,
                                          detail='Stop Loss')
                        elif TradingService.check_take_profit(p, values.high):
                            ts.close_long(asset=asset, day=simulation_day, close_price=p.take_profit, position=p,
                                          detail='Take Profit')
                        elif predicted == SELL:
                            ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                          detail='Sell Signal')
                        elif predicted == HOLD and p_age > 86400 * 3:
                            ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                          detail='Age')
                        elif predicted == BUY:
                            if change > 0:
                                ts.update_stop_loss(asset=asset, position=p, close_price=values.close, pct=-0.05)
                    elif p.type == 'MARGIN_SHORT':
                        if TradingService.check_stop_loss(p, values.high):
                            ts.close_short(asset=asset, day=simulation_day, close_price=p.stop_loss, position=p,
                                           detail='Stop Loss')
                        elif TradingService.check_take_profit(p, values.low):
                            ts.close_short(asset=asset, day=simulation_day, close_price=p.take_profit, position=p,
                                           detail='Take Profit')
                        elif predicted == SELL:
                            # If we had some profit and signal is still SELL, book those by lowering stop loss
                            if change < 0:
                                ts.update_stop_loss(asset=asset, position=p, close_price=values.close, pct=0.05)
                        elif predicted == HOLD and p_age > 86400 * 3:
                            ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                           detail='Age')
                        elif predicted == BUY:
                            ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                           detail='Buy Signal')
                except MessageException as e:
                    print(f"Order handling exception: {e.message}")

            try:
                # If prediction is BUY (price will rise) then open a MARGIN LONG position
                if predicted == BUY:
                    ts.open_long(
                        asset=asset,
                        day=simulation_day,
                        close_price=values.close,
                        size=0.1,
                        stop_loss=-0.1,
                        take_profit=0.05
                    )
                # If prediction is SELL (price will drop) open a MARGIN SHORT position
                elif predicted == SELL:
                    ts.open_short(
                        asset=asset,
                        day=simulation_day,
                        close_price=values.close,
                        size=0.1,
                        stop_loss=0.1,
                        take_profit=-0.05
                    )
            except MessageException as e:
                print(f"Order placement exception: {e.message}")

            # If this is the last trading day of the period, close all open positions
            if index.timestamp() == results.index[-1].timestamp():
                print("Last trading day reached, liquidating all positions..")
                open_positions = ts.get_open_positions(asset=asset, day=simulation_day)
                for p in open_positions:
                    try:
                        if p.type == 'MARGIN_LONG':
                            ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                          detail='Liquidation')
                        elif p.type == 'MARGIN_SHORT':
                            ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                           detail='Liquidation')
                    except MessageException as e:
                        print(f"Order liquidation exception: {e.message}")

            # Update equity value for the asset
            ts.update_equity(asset=asset, day=simulation_day, price=values.close)
            # Update baseline values for the asset
            ts.update_baseline(asset=asset, day=simulation_day, name='buy_and_hold', value=values.close * bh_amount)

        print("Timeframe done.")


if __name__ == '__main__':
    typer.run(main)
