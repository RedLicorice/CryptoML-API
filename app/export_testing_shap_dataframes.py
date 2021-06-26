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
            print(f"Loading estimators for test {t.window}")
            estimators = ModelService.load_test_estimators(model=model, mt=t)
            shaps = []
            print(f"Calculating shap values...")
            for est in tqdm(estimators):
                est_class = y_all.loc[est.day]
                shap_v, shap_exp = get_shap_values(estimator=est, X=X_all.loc[est.day], X_train=est.train_x, bytes=False)
                df = pd.DataFrame([shap_v], index=[pd.to_datetime(est.day)], columns=X_all.columns)
                df['label'] = y_all.loc[est.day]
                df['shap_expected'] = shap_exp
                shaps.append(df)
            print("Exporting dataframe..")
            cdf = pd.concat(shaps, axis='index')
            os.makedirs(f"data/shap_values/{dataset}/{target}/{pipeline}/", exist_ok=True)
            cdf.to_csv(f"data/shap_values/{dataset}/{target}/{pipeline}/shap_test_{symbol}_Wdays{t.window['days']}.csv", index_label='time')
            print("Exported.")
            # # Load day estimator
            # est = load_estimator()

        print(f"Plotted {symbol}")


if __name__ == '__main__':
    typer.run(main)
