import requests
import json

symbols = [
            "ADAUSD", "BCHUSD", "BNBUSD",
            "BTCUSD", "BTGUSD", "DASHUSD",
            "DOGEUSD", "EOSUSD", "ETCUSD",
            "ETHUSD", "LINKUSD", "LTCUSD",
            "NEOUSD", "QTUMUSD", "TRXUSD",
            "USDTUSD", "VENUSD", "WAVESUSD",
            "XEMUSD", "XMRUSD", "XRPUSD",
            "ZECUSD", "ZRXUSD"
        ]

results = []
for symbol in symbols:
    query = {
        "type": "FEATURES",
        "symbol": symbol
    }
    results.append({
        'query': query,
        'symbol': symbol,
        'name': 'merged'
    })
    #res = requests.post('http://127.0.0.1:8000/datasets/merge/merged/{}'.format(symbol), json=query)
    print("{} OK".format(symbol))
print(json.dumps(results, indent=4))
