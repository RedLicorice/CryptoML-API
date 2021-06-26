import typer
from tqdm import tqdm
import os
from cryptoml.util.shap import get_shap_values
import pandas as pd

from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.services.model_service import ModelService

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

def main(dataset: str, target: str, pipeline: str):
    shapes = []
    ds_service = DatasetService()
    m_service = ModelService()
    for symbol in SYMBOLS:
        print(f"Exporting shap dataframes for symbol {symbol}")
        ds = ds_service.get_dataset(name=dataset, symbol=symbol)
        fs = DatasetService.get_feature_selection(ds=ds, method='importances_shap', target=target)
        X_all = ds_service.get_dataset_features(
            ds=ds,
            columns=fs.features
        )
        y_all = ds_service.get_dataset_target(ds=ds, name=target)
        model = m_service.get_model(pipeline=pipeline, dataset=dataset, target=target, symbol=symbol)
        for t in model.tests:
            os.makedirs(f"data/shap_values/{dataset}/{target}/{pipeline}/", exist_ok=True)
            placeholder= "{label}"
            csv_name = f"data/shap_values/{dataset}/{target}/{pipeline}/shap_training_window_{symbol}_{placeholder}_Wdays{t.window['days']}_.csv"
            print(f"Loading estimators for test {t.window}")
            estimators = ModelService.load_test_estimators(model=model, mt=t)
            results = ModelService.parse_test_results(test=t)
            shaps = [[], [], []]

            X_test = X_all.loc[t.test_interval.begin:t.test_interval.end]
            shap_expected = []
            print(f"Calculating shap values...")
            for est in tqdm(estimators):
                est_class = y_all.loc[est.day]
                shap_v, shap_exp = get_shap_values(estimator=est.named_steps.c, X=est.train_x, bytes=False)
                shap_expected.append([est.day] + [v for v in shap_exp])
                for cls, label in enumerate(["SELL", "HOLD", "BUY"]):
                    df = pd.DataFrame(shap_v[cls], index=est.train_x.index, columns=est.train_x.columns)
                    if not shaps[cls]: # If list is empty, append whole df
                        shaps[cls].append(df)
                    else:
                        shaps[cls].append(df.iloc[-1:]) # otherwise only append new row (sliding window)

            for cls, label in enumerate(["SELL", "HOLD", "BUY"]):
                class_csv_name = csv_name.format(label=label)
                print(f"Exporting dataframe for class {label} -> {class_csv_name}")
                cdf = pd.concat(shaps[cls], axis='index')
                cdf.to_csv(class_csv_name, index_label='time')

            expected_csv_name = csv_name.format(label='SHAP_expected')
            print(f"Exporting expected values dataframe -> {expected_csv_name}")
            edf = pd.DataFrame(
                shap_expected,
                columns=["time", "shap_expected_sell", "shap_expected_hold", "shap_expected_buy"],
            )
            edf.to_csv(expected_csv_name, index_label='time')

            print(f"Exported symbol {symbol}.")
            # # Load day estimator
            # est = load_estimator()

        print(f"Plotted {symbol}")


if __name__ == '__main__':
    typer.run(main)
