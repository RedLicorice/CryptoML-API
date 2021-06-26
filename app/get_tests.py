import typer
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.models.classification import Model, ModelTest
from cryptoml_core.util.timestamp import get_timestamp, from_timestamp
from cryptoml_core.exceptions import MessageException
import json
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging
import pandas as pd
import numpy as np
from matplotlib import cm
from typing import List

symbols = [
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
    models = ModelService()
    writer = pd.ExcelWriter("{}_{}.xlsx".format(dataset, target))
    for symbol in symbols:
        model_tests = models.compare_models(symbol, dataset, target)
        result = []
        for m in model_tests:
            test = m["tests"]
            report = test["classification_report"] if "classification_report" in test else None
            #duration = from_timestamp(test["end_at"]).timestamp() - from_timestamp(test["start_at"]).timestamp()
            # mean_dur = pd.Series(test["classification_results"]["duration"]).mean()
            results = ModelService.parse_test_results(test)
            result.append({
                "pipeline": m["pipeline"],
                "window": test['window']['days'],
                #"step": str(test["step"]),
                "mean_fit_time": results.fit_time.mean(),
                "mean_predict_time": results.predict_time.mean(),
                "support_all": report["total_support"] if report else np.nan,
                "support_0": report["sup_0"] if report else np.nan,
                "support_1": report["sup_1"] if report else np.nan,
                "support_2": report["sup_2"] if report else np.nan,
                # Per-class precision/recall/f1/spe/geom/iba
                "precision_0": report["pre_0"] if report else np.nan,
                "recall_0": report["rec_0"] if report else np.nan,
                "specificity_0": report["spe_0"] if report else np.nan,
                "f1-score_0": report["f1_0"] if report else np.nan,
                "geometric_mean_0": report["geo_0"] if report else np.nan,
                "index_balanced_accuracy_0": report["iba_0"] if report else np.nan,

                "precision_1": report["pre_1"] if report else np.nan,
                "recall_1": report["rec_1"] if report else np.nan,
                "specificity_1": report["spe_1"] if report else np.nan,
                "f1-score_1": report["f1_1"] if report else np.nan,
                "geometric_mean_1": report["geo_1"] if report else np.nan,
                "index_balanced_accuracy_1": report["iba_1"] if report else np.nan,

                "precision_2": report["pre_2"] if report else np.nan,
                "recall_2": report["rec_2"] if report else np.nan,
                "specificity_2": report["spe_2"] if report else np.nan,
                "f1-score_2": report["f1_2"] if report else np.nan,
                "geometric_mean_2": report["geo_2"] if report else np.nan,
                "index_balanced_accuracy_2": report["iba_2"] if report else np.nan,

                # Roc-auc
                # "roc_auc_ovo_macro": report["roc_auc_ovo_macro"] if report else np.nan,
                # "roc_auc_ovo_weighted": report["roc_auc_ovo_weighted"] if report else np.nan,
                # "roc_auc_ovr_macro": report["roc_auc_ovr_macro"] if report else np.nan,
                # "roc_auc_ovr_weighted": report["roc_auc_ovr_weighted"] if report else np.nan,
                # Averages
                "precision_avg": report["avg_pre"] if report else np.nan,
                "recall_avg": report["avg_rec"] if report else np.nan,
                "specificity_avg": report["avg_spe"] if report else np.nan,
                "f1-score_avg": report["avg_f1"] if report else np.nan,
                "geometric_mean_avg": report["avg_geo"] if report else np.nan,
                "index_balanced_accuracy_avg": report["avg_iba"] if report else np.nan,

            })
        df = pd.DataFrame(result)
        # Plot to XLSX with conditional formatting by coolwarm color map,
        # ordering by ascending accuracy
        df.sort_values(by='precision_avg', ascending=True)\
            .style.background_gradient(cmap=cm.get_cmap('coolwarm')) \
            .format(None, na_rep="-")\
            .to_excel(writer, sheet_name=symbol, index_label="#", float_format = "%0.3f")
        # Adjust column width
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column) + 1
            writer.sheets[symbol].set_column(col_idx, col_idx, column_length)
    writer.close()



if __name__ == '__main__':
    setup_file_logger('get_tests.log')
    typer.run(main)
