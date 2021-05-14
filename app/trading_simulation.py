import typer

from cryptoml_core.exceptions import MessageException
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.services.trading_service import TradingService
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.util.timestamp import to_timestamp

import numpy as np
import pandas as pd

SELL, HOLD, BUY = range(3)
TARGETS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}


def main(pipeline: str, dataset: str, symbol: str):
    ds = DatasetService()
    ms = ModelService()
    ts = TradingService()
    ohlcv_ds = ds.get_dataset('ohlcv', symbol=symbol)
    ohlcv = ds.get_dataset_features(ohlcv_ds)[ohlcv_ds.valid_index_min:ohlcv_ds.valid_index_max]

    model = ms.get_model(pipeline, dataset, 'class', symbol)
    for test in model.tests:
        # Re-convert classification results from test to a DataFrame
        results = pd.DataFrame(test.classification_results)
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        results.index = pd.to_datetime(results.index)

        asset = ts.get_asset(pipeline=pipeline, dataset=dataset, target='class', symbol=symbol, window=test.window['days'])
        # Now use classification results to trade!
        for index, pred in results.iterrows():
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
            _index = ohlcv.index.get_loc(index)
            change = (values['close'] - values['open']) / values['open']

            print(f"Day [{index}] "
                  f"[O {values.open} H {values.high} L {values.low} C {values.close}] "
                  f"PCT={change}% "
                  f"LABEL={TARGETS[label]} PRED={TARGETS[predicted]}"
                  )
            open_positions = ts.get_open_positions(asset=asset, day=simulation_day)
            for p in open_positions:
                p_age = TradingService.get_position_age(position=p, day=simulation_day)
                if p.type == 'MARGIN_LONG':
                    if TradingService.check_stop_loss(p, values.low):
                        ts.close_long(asset=asset, day=simulation_day, close_price=p.stop_loss, position=p, detail='Stop Loss')
                    elif TradingService.check_take_profit(p, values.high):
                        ts.close_long(asset=asset, day=simulation_day, close_price=p.take_profit, position=p, detail='Take Profit')
                    elif predicted == SELL:
                        ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p, detail='Sell Signal')
                    elif predicted == HOLD and p_age > 86400*3:
                        ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p, detail='Age')
                    elif predicted == BUY and change > 0:
                        ts.update_stop_loss(asset=asset, position=p, close_price=values.close, pct=-0.05)
                elif p.type == 'MARGIN_SHORT':
                    if TradingService.check_stop_loss(p, values.high):
                        ts.close_short(asset=asset, day=simulation_day, close_price=p.stop_loss, position=p, detail='Stop Loss')
                    elif TradingService.check_take_profit(p, values.low):
                        ts.close_short(asset=asset, day=simulation_day, close_price=p.take_profit, position=p, detail='Take Profit')
                    elif predicted == SELL and change < 0:
                        # If we had some profit and signal is still SELL, book those by lowering stop loss
                        ts.update_stop_loss(asset=asset, position=p, close_price=values.close, pct=0.05)
                    elif predicted == HOLD and p_age > 86400*3:
                        ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p, detail='Age')
                    elif predicted == BUY:
                        ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p, detail='Buy Signal')

            # If prevision is BUY (price will rise) then open a MARGIN LONG position
            if predicted == BUY:
                ts.open_long(asset=asset, day=simulation_day, close_price=values.close, size=0.1, stop_loss=-0.1, take_profit=0.05)
            # If prediction is SELL (price will drop) open a MARGIN SHORT position
            elif predicted == SELL:
                ts.open_short(asset=asset, day=simulation_day, close_price=values.close, size=0.1, stop_loss=0.1, take_profit=-0.03)

            # If this is the last trading day of the period, close all open positions
            if index == results.index[-1]:
                open_positions = ts.get_open_positions(asset=asset, day=simulation_day)
                for p in open_positions:
                    if p.type == 'MARGIN_LONG':
                        ts.close_long(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                      detail='Liquidation')
                    elif p.type == 'MARGIN_SHORT':
                        ts.close_short(asset=asset, day=simulation_day, close_price=values.close, position=p,
                                       detail='Liquidation')
            # Update equity value for the asset
            ts.update_equity(asset=asset, day=simulation_day, price=values.close)

        print("Timeframe done")


if __name__ == '__main__':
    typer.run(main)
