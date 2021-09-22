import typer
from cryptoml_core.services.dataset_service import DatasetService
import numpy as np
import pandas as pd
from cryptoml_core.util.timestamp import from_timestamp
import plotly.express as px
import plotly.graph_objects as go

SYMBOLS = [
    "ADAUSD", "BCHUSD", "BNBUSD",
    "BTCUSD", "BTGUSD", "DASHUSD",
    "DOGEUSD", "EOSUSD", "ETCUSD",
    "ETHUSD", "LINKUSD", "LTCUSD",
    "NEOUSD", "QTUMUSD", "TRXUSD",
    "WAVESUSD", "XEMUSD",
    "XMRUSD", "XRPUSD", "ZECUSD",
    "ZRXUSD"
] # "VENUSD",


def main(dataset: str):
    dss = DatasetService()
    records = []
    for symbol in SYMBOLS:
        ds = dss.get_dataset(name=dataset, symbol=symbol)
        fs = DatasetService.get_feature_selection(ds, 'importances_shap', 'class')
        target = dss.get_dataset_target(ds=ds, name='class')
        uniq, cnt = np.unique(target, return_counts=True)
        if cnt[0]+cnt[1]+cnt[2] != ds.count:
            print(f"Mismatch between classes and count in {symbol}")
        mindt = from_timestamp(ds.valid_index_min)
        maxdt = from_timestamp(ds.valid_index_max)
        daysn = (maxdt - mindt).days
        records.append({
            'Pair': symbol,
            'num_features': len(ds.features),
            'sel_features': len(fs.features),
            'min_index': ds.valid_index_min,
            'max_index': ds.valid_index_max,
            'valid_days': daysn,
            'records': ds.count,
            'sell_count': cnt[0],
            'hold_count': cnt[1],
            'buy_count': cnt[2]
        })
    df = pd.DataFrame.from_records(records)
    fig = px.timeline(df, x_start="min_index", x_end="max_index", y="Pair")
    fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    #fig.show()
    fig.update_layout(
        title={
            'text': f"Sample distribution across datasets",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.5)',
        font={'color': 'White'},
        margin={
            'l':5, 'r': 5, 't': 80, 'b': 5, 'pad': 5
        }
    )
    fig.write_image("images/data_summary/timeline.png")
    for symbol in SYMBOLS:
        sdf = df[df.Pair == symbol]
        pie_values = [sdf['sell_count'].values[0], sdf['hold_count'].values[0], sdf['buy_count'].values[0]]
        pie_labels = ['SELL', 'HOLD', 'BUY']
        sfig = go.Figure(data=[
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                textinfo='label+percent',
                #insidetextorientation='radial',
                showlegend=False
            )
        ])
        sfig.update_layout(
            title={
                'text': f"Class distribution for pair {symbol}",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 22
                }
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={
                'color':'White',
                'size': 26
            },
            margin={
                'l':0, 'r': 0, 't': 80, 'b': 0, 'pad': 0
            },
            uniformtext_minsize=24
        )

        sfig.write_image(f"images/data_summary/{symbol}_distribution.png")
    print(df.head())


if __name__ == '__main__':
    typer.run(main)