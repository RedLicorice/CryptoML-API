import typer
from cryptoml.util.shap import shap
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
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

def get_mean_df(dataset: str, target:str):
    dfs = [
        (symbol, pd.read_csv(f"data/selection_{dataset}_{target}/{symbol}_feature_importances_shap.csv", index_col='index').dropna(how='any', axis='index'))
        for symbol in SYMBOLS
    ]

    df_mean = pd.DataFrame()
    for symbol, df in dfs:
        df_mean[f'importances_{symbol}'] = df.importance

    df_base = dfs[0][1][['category', 'subgroup', 'name']].copy()
    df_base['importance'] = df_mean.mean(axis='columns', numeric_only=True)
    return df_base


def df_to_faceted_figure(df, show=False, symbol=None):
    if symbol:
        symbol = symbol + " "
    else:
        symbol = ""
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.20,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            f"{symbol}Importance hierarchy",
            None,
            "Importance by Category",
            "Importance by Subcategory"
        ],
        specs=[
            [{"type": "pie", "colspan": 2}, {}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )

    # Bit of a cheat: use plotly express to do the data preprocessing for us, then render using go!
    fig_sunburst = px.sunburst(df, path=['category', 'subgroup', 'name'], values='importance', title="Mean XGBoost feature importances across analyzed currencies")
    # fig_sunburst.show()

    sunburst = go.Sunburst(
        labels=fig_sunburst['data'][0]['labels'].tolist(),
        parents=fig_sunburst['data'][0]['parents'].tolist(),
        values=fig_sunburst['data'][0]['values'].tolist(),
        ids=fig_sunburst['data'][0]['ids'].tolist(),
        branchvalues='total',
        insidetextorientation='radial'
    )
    fig.add_trace(sunburst, row=1, col=1)

    category_df = df.groupby(by='category').sum()
    subgroup_df = df.groupby(by='subgroup').sum()

    cat_bar = go.Bar(
        x=category_df.importance,
        y=category_df.index,
        orientation='h',
        showlegend=False,
        marker_color='blue'
    )
    fig.add_trace(cat_bar, row=2, col=1)

    sub_bar = go.Bar(
        x=subgroup_df.importance,
        y=subgroup_df.index,
        orientation='h',
        showlegend=False,
        marker_color='blue'
    )
    fig.add_trace(sub_bar, row=2, col=2)

    fig.update_yaxes(**{
        'dtick': 1,
        'type': 'category'
    })
    if show:
        fig.show()

    fig.update_layout(
        autosize=False,
        margin=dict(
            l=30,
            r=30,
            b=50,
            t=25,
            pad=4
        )
    )
    return fig

def df_to_barh(df, show=False, symbol=None):
    fig = go.Figure()

    df = df.sort_values(by='importance', ascending=True)[df.importance > 0]

    imp_bar = go.Bar(
        x=df.importance,
        y=df.name,
        # marker=dict(color=df.category.replace()),
        orientation='h'
    )
    fig.add_trace(imp_bar)

    fig.update_yaxes(**{
        'dtick': 1,
        'type': 'category'
    })
    # fig.update_xaxes(**{
    #     'type': 'log'
    # })

    if show:
        fig.show()


def main(dataset: str, target: str):
    # mean_df = get_mean_df(dataset, target)
    # fig = df_to_faceted_figure(mean_df, False)
    # fig.write_image("images/feature_selection/feature_selection_mean.png", width=800, height=1024, scale=1)

    for symbol in SYMBOLS:
        symbol_df = pd.read_csv(
            f"data/selection_{dataset}_{target}/{symbol}_feature_importances.csv",
            index_col='index'
        ).dropna(how='any', axis='index')

        symbol_selected_df = pd.read_csv(
            f"data/selection_{dataset}_{target}/{symbol}_feature_importances_selected.csv",
            index_col='index'
        ).dropna(how='any', axis='index')

        # fig = df_to_faceted_figure(symbol_df, False, symbol)
        # fig.write_image(f"images/feature_selection/importances_{symbol}.png", width=800, height=1024, scale=1)

        # fig = df_to_faceted_figure(symbol_selected_df, False, symbol)
        # fig.write_image(f"images/feature_selection/selected_{symbol}.png", width=800, height=1024, scale=1)
        df_to_barh(symbol_selected_df, True)

        print("Plotted")

if __name__ == '__main__':
    typer.run(main)
