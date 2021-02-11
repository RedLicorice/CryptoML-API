import krakenex
#from ..config import config
from time import time_ns
import pandas as pd
import time

class KrakenService:
    def __init__(self):
        self.kraken = krakenex.API()
        # keypath = config['kraken']['key'].get(str)
        # if keypath:
        #     self.kraken.load_key(keypath)

    # Gets tick data from Kraken API, kind of slow (ie, not suitable for bootstrapping)
    def get_historical_trades(self, symbol, **kwargs):
        last = kwargs.get('begin', 0)
        end = kwargs.get('end', time_ns())
        # All columns included in API Response
        columns = ['price', 'volume', 'time', 'buy_sell', 'market_limit', 'miscellaneous']
        # Initialize the result dataframe with data we're interested in
        result = pd.DataFrame(columns=columns[:-3])
        while last < end:
            print("Querying Kraken Trades: LAST={}, BEGIN={}".format(last, end))
            response = self.kraken.query_public('Trades', {'pair':symbol, 'since': last})
            last = int(response['result']['last'])
            data = response['result'][symbol]
            # Create a temporary dataframe for response data
            df = pd.DataFrame(data, columns=columns)
            # Drop columns we're not interested in
            df = df.drop(columns[3:], axis='columns')
            # Append temporary dataframe to result dataframe
            result = result.append(df)
            # Sleep 3s: this suffices not to hit the API Limit (10 requests every 30 seconds, earn 10 points spend 20)
            time.sleep(3)
        # Parse time column from floating point nanosecond timestamp to
        # datetime index
        result.time = pd.to_datetime(result.time, unit='s')
        result.set_index('time')
        # Return resulting dataframe
        return result

if __name__ == '__main__':
    begin = time_ns() - ( 120 * 1000000000) #  30 Days * 86400S/Day * 1000000000 nS/S
    k = KrakenService()
    d = k.get_historical_trades('XXBTZUSD', begin=begin)
    print(d)