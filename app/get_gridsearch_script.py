import typer
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml.pipelines import PIPELINE_LIST
from typing import Optional


def main(dataset: str, target: str, splits: Optional[int] = 1, type: Optional[str] = 'sh'):
    dss = DatasetService()
    symbols = dss.get_dataset_symbols(name=dataset)

    lines = []
    for symbol in symbols:
        for pipeline in PIPELINE_LIST:
            lines.append(f"python grid_search_new.py {symbol} {dataset} {target} {pipeline} --feature-selection-method importances_shap")

    destfile = f"gridsearch_{dataset}_{target}_all"
    if type == 'cmd':
        with open(destfile + ".cmd", "w") as f:
            f.write("\n".join(["@echo off"] + lines))
    elif type == 'sh':
        with open(destfile + ".sh", "w") as f:
            f.write("\n".join(["#!/bin/bash"] + lines))
    print(f"Grid search script saved to {destfile}")
    return destfile


if __name__ == '__main__':
    typer.run(main)