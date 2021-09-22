import pandas as pd
import typer
import plotly.graph_objects as go

SYMBOLS = [
    "ADAUSD", "BCHUSD", "BNBUSD",
    "BTCUSD", "BTGUSD", "DASHUSD",
    "DOGEUSD", "EOSUSD", "ETCUSD",
    "ETHUSD", "LINKUSD", "LTCUSD",
    "NEOUSD", "QTUMUSD", "TRXUSD",
    "WAVESUSD", "XEMUSD",
    "XMRUSD", "XRPUSD", "ZECUSD",
    "ZRXUSD",
    # "VENUSD"
]

PIPELINE_REPL = {
    # 'adaboost_decisiontree',
    'bagging_decisiontree': 'Bagging + DT',
    # 'bagging_linear_svc',
    'bagging_poly_svc': 'Bagging + SVC (Poly)',
    # 'bagging_rbf_svc',
    # 'debug_xgboost',
    'plain_knn': 'k-NN',
    # 'plain_linear_svc',
    'plain_mlp': 'MLP',
    #'plain_mlp_big',
    'plain_mnb': 'MNB',
    'plain_poly_svc': 'SVC (Poly)',
    'plain_randomforest': 'RandomForest',
    # 'plain_rbf_svc',
    'plain_xgboost': 'XGBoost',
    'smote_knn': 'SMOTE + k-NN',
    'smote_mlp': 'SMOTE + MLP',
    'smote_poly_svc': 'SMOTE + SVC (Poly)'
}

def main(filename):
    dfs = []
    for symbol in SYMBOLS:
        df = pd.read_excel(io=filename, sheet_name=symbol, header=0, index_col='pipeline')
        rdf = df[[
            'window',
            #'support_0', 'support_1', 'support_2',
            'precision_0', 'precision_1', 'precision_2',
            'precision_avg', 'index_balanced_accuracy_avg'
        ]]
        rdf = rdf.sort_values(by='precision_avg', ascending=False)[:3]
        rdf.insert(0, 'symbol', symbol)
        dfs.append(rdf)
    all_df = pd.concat(dfs).sort_values(by='precision_avg', ascending=False)
    all_df.insert(0, 'algorithm', all_df.index)
    all_df.index = pd.Index(range(1, all_df.shape[0]+1))
    all_tex = all_df.to_latex()
    print("pause")
    all_df = all_df.head(10)
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Algorithm', 'W', 'SELL', 'HOLD', 'BUY', 'PRE', 'IBA'],
                    fill_color='turquoise',
                    align='left'),
        cells=dict(values=[all_df.algorithm.replace(PIPELINE_REPL), all_df.window, all_df.precision_0, all_df.precision_1, all_df.precision_2, all_df.precision_avg, all_df.index_balanced_accuracy_avg],
                   #fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(
        title={
            'text': f"Model test results",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.5)',
        font={'color': 'White'},
        margin={
            'l': 5, 'r': 5, 't': 100, 'b': 5, 'pad': 5
        }
    )
    fig.write_image("images/model_test_summary.png")




if __name__ == '__main__':
    typer.run(main)