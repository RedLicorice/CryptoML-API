import typer
from cryptoml.util.shap import shap
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import os
import shap
import numpy as np
from matplotlib import pyplot as plt

from cryptoml_core.services.dataset_service import DatasetService
from cryptoml.util.shap import parse_shap_values
# from plot_feature_selection_hierarchy import load_hierarchy

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

def main(dataset: str, target: str):
    # hierarchy = load_hierarchy(f"{dataset}_{target}_feature_hierarchy.yml")
    # hdf = pd.DataFrame(hierarchy)

    shapes = []
    for symbol in SYMBOLS:
        ds_service = DatasetService()
        ds = ds_service.get_dataset(name=dataset, symbol=symbol)
        fs = DatasetService.get_feature_selection(ds=ds, method='importances_shap', target=target)
        shap_v, shap_exp = parse_shap_values(fs.shap_values)

        X_train = ds_service.get_dataset_features(
            ds=ds,
            begin=fs.search_interval.begin,
            end=fs.search_interval.end#,
            #columns=fs.features
        )
        shapes.append(X_train.shape[0])

        shap_0 = pd.DataFrame(shap_v[0], index=X_train.index, columns=X_train.columns)
        shap_1 = pd.DataFrame(shap_v[1], index=X_train.index, columns=X_train.columns)
        shap_2 = pd.DataFrame(shap_v[2], index=X_train.index, columns=X_train.columns)

        sel_train = X_train[fs.features]
        sel_shap_0 = shap_0[fs.features]
        sel_shap_1 = shap_1[fs.features]
        sel_shap_2 = shap_2[fs.features]

        show_count = 50 #len(fs.features)
        shap.summary_plot(sel_shap_0.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class SELL")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_SELL_top{show_count}.png")
        plt.close()

        shap.summary_plot(sel_shap_1.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class HOLD")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_HOLD_top{show_count}.png")
        plt.close()

        shap.summary_plot(sel_shap_2.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class BUY")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_BUY_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_0.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class SELL")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_SELL_abs_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_1.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class HOLD")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_HOLD_abs_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_2.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class BUY")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_BUY_abs_top{show_count}.png")
        plt.close()

        show_count = 25
        shap.summary_plot(sel_shap_0.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class SELL")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_SELL_top{show_count}.png")
        plt.close()

        shap.summary_plot(sel_shap_1.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class HOLD")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_HOLD_top{show_count}.png")
        plt.close()

        shap.summary_plot(sel_shap_2.values, sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"SHAP Summary plot for {symbol}, top {show_count} features for class BUY")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_BUY_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_0.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class SELL")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_SELL_abs_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_1.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class HOLD")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_HOLD_abs_top{show_count}.png")
        plt.close()

        shap.summary_plot(np.abs(sel_shap_2.values), sel_train, max_display=show_count, show=False)
        plt.tight_layout()
        plt.title(f"Absolute SHAP Summary plot for {symbol}, top {show_count} features for class BUY")
        plt.savefig(f"images/shap-global/{dataset}_{target}__shap__summary_plot_{symbol}_BUY_abs_top{show_count}.png")
        plt.close()
        
        print(f"Plotted {symbol}")


if __name__ == '__main__':
    typer.run(main)
