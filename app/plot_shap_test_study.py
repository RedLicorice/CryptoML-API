import typer
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
import shap
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.model_service import ModelService
from imblearn.metrics import classification_report_imbalanced
from datetime import timedelta
import math

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

PIPELINES = [
    # 'bagging_decisiontree',
    # 'bagging_poly_svc',
    # 'plain_knn',
    # 'plain_mlp',
    # 'plain_mnb',
    # 'plain_poly_svc',
    # 'plain_randomforest',
    'plain_xgboost',
    # 'smote_knn',
    # 'smote_mlp',
    # 'smote_poly_svc'
]

def get_metrics_df(results: pd.DataFrame):
    cri_records = []
    for i in range(1, results.shape[0]):
        cri = classification_report_imbalanced(
            y_true=results.label.iloc[:i],
            y_pred=results.predicted.iloc[:i],
            output_dict=True,
            zero_division=0
        )
        cri_record = {}
        for k, v in cri.items():
            if isinstance(v, dict):
                for m, s in v.items():
                    cri_record[f"{m}_{k}"] = s
            else:
                cri_record[f"{m}"] = v
        cri_record['time'] = results.time.iloc[i]
        cri_records.append(cri_record)
    cri_df = pd.DataFrame(cri_records)
    cri_df.index = pd.to_datetime(cri_df.time)
    return cri_df

def main(dataset: str, target: str):
    num_shap_plots = 3
    shap_show_count = 10

    ds_service = DatasetService()
    m_service = ModelService()
    for pipeline in PIPELINES:
        for symbol in SYMBOLS:
            print(f"Plotting shap dataframes for pipeline {pipeline} symbol {symbol}")
            ds = ds_service.get_dataset(name=dataset, symbol=symbol)
            fs = DatasetService.get_feature_selection(ds=ds, method='importances_shap', target=target)
            X_all = ds_service.get_dataset_features(
                ds=ds,
                columns=fs.features
            )
            y_all = ds_service.get_dataset_target(ds=ds, name=target)
            model = m_service.get_model(pipeline=pipeline, dataset=dataset, target=target, symbol=symbol)
            for t in model.tests:
                placeholder= "{label}"
                csv_name = f"data/shap_values/{dataset}/{target}/{pipeline}/shap_training_window_{symbol}_{placeholder}_Wdays{t.window['days']}_.csv"
                expected_csv_name = csv_name.format(label='SHAP_expected')
                print(f"Loading results for test {t.window}")
                results = ModelService.parse_test_results(test=t)
                exp_shap_df = pd.read_csv(expected_csv_name, index_col='time', parse_dates=True)
                for cls, label in enumerate(["SELL", "HOLD", "BUY"]):
                    class_csv_name = csv_name.format(label=label)
                    cls_shap_df = pd.read_csv(class_csv_name, index_col='time', parse_dates=True)
                    cls_shap_df = cls_shap_df.loc[t.test_interval.begin: t.test_interval.end]

                    x_train = X_all.loc[cls_shap_df.index]
                    chunk_size = int(cls_shap_df.shape[0] / num_shap_plots)

                    fig = plt.figure(constrained_layout=True, figsize=(100, 50), dpi=300) #
                    gs = GridSpec(3, num_shap_plots, figure=fig, wspace=1.5, hspace=0.3)
                    precision_ax = fig.add_subplot(gs[0, :])
                    shap_values_ax = fig.add_subplot(gs[1, :])
                    beeswarms_axs = [fig.add_subplot(gs[2, i]) for i in range(num_shap_plots)]
                    #format_axes(fig)
                    shap_plot_labels = set()
                    first_shap_day = results.iloc[0]['time'].replace('+00:00', '').replace('T', '').replace(':', '').replace('-', '')
                    middle_shap_day = results.iloc[int(results.shape[0] / 2)]['time'].replace('+00:00', '').replace('T', '').replace(':', '').replace('-', '')
                    last_shap_day = results.iloc[-1]['time'].replace('+00:00', '').replace('T', '').replace(':', '').replace('-', '')
                    for idx, dayname in enumerate([first_shap_day, middle_shap_day, last_shap_day]):
                        day_csv_name = f"data/shap_values/{dataset}/{target}/{pipeline}/daily/shap_training_window_{symbol}_{label}_Wdays{t.window['days']}_DAY{dayname}.csv"

                        # Plot each section's SHAP values
                        cdf_subset = pd.read_csv(day_csv_name, index_col='time', parse_dates=True)
                        train_subset = X_all.loc[cdf_subset.index]

                        # Get a rank of feature labels based on this section's shap values
                        abs_mean_shap = cdf_subset.abs().mean(axis='index')
                        abs_mean_rank = abs_mean_shap.sort_values(ascending=False)[:shap_show_count]
                        for l in abs_mean_rank.index:
                            # Save labels for features in the top-N
                            shap_plot_labels.add(l)

                        # Plot this section's SHAP values
                        plt.sca(beeswarms_axs[idx])
                        shap.summary_plot(
                            cdf_subset.values,
                            train_subset,
                            max_display=shap_show_count,
                            show=False,
                            color_bar=False,
                            sort=True
                        )
                        min_date = cdf_subset.index.min().to_pydatetime()
                        max_date = cdf_subset.index.max().to_pydatetime() + timedelta(days=1)
                        min_date_f = min_date.strftime("%Y/%m/%d")
                        max_date_f = max_date.strftime("%Y/%m/%d")
                        beeswarms_axs[idx].set_xlabel(f"SHAP values\nWindow: {min_date_f} - {max_date_f}", fontsize=8)
                        beeswarms_axs[idx].tick_params(axis='y', which='major', labelsize=6)
                        beeswarms_axs[idx].tick_params(axis='x', which='major', labelsize=8)

                    # Plot shap values
                    day_csv_name = f"data/shap_values/{dataset}/{target}/{pipeline}/shap_training_window_{symbol}_{label}_Wdays{t.window['days']}_.csv"
                    plot_cls_shap_df = pd.read_csv(day_csv_name, index_col='time', parse_dates=True)
                    def get_spread(series):
                        return np.abs(series.max() - series.min())
                    plot_rank = plot_cls_shap_df[list(shap_plot_labels)].apply(get_spread, axis='index').sort_values(ascending=False)[:shap_show_count]
                    plot_cls_shap_df['xlabel'] = [t.to_pydatetime().strftime("%Y/%m/%d") for t in plot_cls_shap_df.index]
                    shap_ax = plot_cls_shap_df.plot(
                        x='xlabel',
                        y=[c for c in plot_rank.index],
                        kind='line',
                        ax=shap_values_ax,
                        legend=False,
                        xlabel=''
                    )
                    patches, labels = shap_ax.get_legend_handles_labels()
                    shap_ax.legend(
                        patches, labels,
                        loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}
                    )
                    shap_ax.tick_params(axis='x', which='major', labelsize=8)
                    shap_ax.set_ylabel('mean(|SHAP|)', fontsize=6)
                    #shap_ax.tick_params(labelbottom=False, labelleft=False)

                    # Get Metrics scores dataframe
                    cri_df = get_metrics_df(results).rolling(7, min_periods=1).mean()
                    cri_df['xlabel'] = [t.to_pydatetime().strftime("%Y/%m/%d") for t in cri_df.index]
                    cri_ax = cri_df.plot(
                        x='xlabel',
                        y=f"pre_{cls}",
                        kind='line',
                        ax=precision_ax,
                        legend=False,
                        xlabel=''
                    )
                    patches, labels = cri_ax.get_legend_handles_labels()
                    cri_ax.legend(
                        patches, labels,
                        loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}
                    )
                    cri_ax.set_ylabel('mean(precision)', fontsize=6)
                    cri_ax.tick_params(labelbottom=False, labelleft=True)

                    min_date = cri_df.index.min().to_pydatetime().strftime("%Y/%m/%d")
                    max_date = cri_df.index.max().to_pydatetime().strftime("%Y/%m/%d")
                    window = t.window['days']
                    fig.suptitle(f"{symbol}, {pipeline}, W={window}D, Class {label}, From {min_date} to {max_date}")

                    # fig.show()
                    os.makedirs(f"images/shap-test-final/", exist_ok=True)
                    plt.savefig(
                        f"images/shap-test-final/{pipeline}_W{window}D_{dataset}_{target}_{symbol}_{label}.png",
                        dpi='figure'
                    )
                    plt.close()
                    print(f"{label} OK")






            print(f"Exported symbol {symbol}.")
            # # Load day estimator
            # est = load_estimator()

        print(f"Plotted {symbol}")


if __name__ == '__main__':
    typer.run(main)
