import typer
import pandas as pd
from cryptoml_core.services.trading_service import TradingService


def main():
    ts = TradingService()
    assets = ts.get_all_assets()
    records = []
    for a in assets:
        records.append({
            'pipeline': a.pipeline,
            'symbol': a.symbol,
            'window': a.window,
            'baseline': a.baselines[-1].equity,
            'equity': a.equities[-1].equity
        })
    df = pd.DataFrame.from_records(records)
    edf = df.sort_values(by='equity', ascending=False).groupby('symbol', sort=False).head(3)
    edf.index = range(1, edf.shape[0] +1)
    tex = edf.to_latex()
    print(df.head())


if __name__ == '__main__':
    typer.run(main)