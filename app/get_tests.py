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


def main(dataset: str, target: str, symbols: List[str]):
    models = ModelService()
    writer = pd.ExcelWriter("{}_{}.xlsx".format(dataset, target))
    for symbol in symbols:
        model_tests = models.compare_models(symbol, dataset, target)
        result = []
        for m in model_tests:
            test = m["tests"]
            report = test["classification_report"] if "classification_report" in test else None
            duration = from_timestamp(test["end_at"]).timestamp() - from_timestamp(test["start_at"]).timestamp()
            result.append({
                "pipeline": m["pipeline"],
                "window": test['window']['days'],
                #"step": str(test["step"]),
                "duration": duration,
                "precision_0": report["precision_0"] if report else np.nan,
                "recall_0": report["recall_0"] if report else np.nan,
                "f1-score_0": report["f1-score_0"] if report else np.nan,
                "support_0": report["support_0"] if report else np.nan,
                "precision_1": report["precision_1"] if report else np.nan,
                "recall_1": report["recall_1"] if report else np.nan,
                "f1-score_1": report["f1-score_1"] if report else np.nan,
                "support_1": report["support_1"] if report else np.nan,
                "precision_2": report["precision_2"] if report else np.nan,
                "recall_2": report["recall_2"] if report else np.nan,
                "f1-score_2": report["f1-score_2"] if report else np.nan,
                "support_2": report["support_2"] if report else np.nan,
                "accuracy": report["accuracy"] if report else np.nan,
                "precision_macro_avg": report["precision_macro_avg"] if report else np.nan,
                "recall_macro_avg": report["recall_macro_avg"] if report else np.nan,
                "f1-score_macro_avg": report["f1-score_macro_avg"] if report else np.nan,
                "precision_weighted_avg": report["precision_weighted_avg"] if report else np.nan,
                "recall_weighted_avg": report["recall_weighted_avg"] if report else np.nan,
                "f1-score_weighted_avg": report["f1-score_weighted_avg"] if report else np.nan,
            })
        df = pd.DataFrame(result)
        # Plot to XLSX with conditional formatting by coolwarm color map,
        # ordering by ascending accuracy
        df.sort_values(by='accuracy', ascending=True)\
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
