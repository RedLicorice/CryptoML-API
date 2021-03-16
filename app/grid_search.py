import typer
from cryptoml_core.services.tuning_service import TuningService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from typing import Optional


def main(dataset: str, target: str, pipeline: str, features: Optional[str] = None):
    tuning = TuningService()
    models = ModelService()
    query = {"dataset": dataset, "target": target, "pipeline": pipeline}
    if pipeline == 'all':
        del query['pipeline']
    if target == 'all':
        del query['target']
    models.clear_parameters(query)
    search_models = models.query_models(query)
    print("[i] {} models to train".format(len(search_models)))
    for i, m in enumerate(search_models):
        print("==[{}/{}]== MODEL: {} {} {} {} =====".format(i+1, len(search_models), m.symbol, m.dataset, m.target, m.pipeline))
        mp = tuning.create_parameters_search(m, split=0.7, features=features)
        print("[{}] Start grid search".format(get_timestamp()))
        mp = tuning.grid_search(m, mp, sync=True, verbose=1, n_jobs=8)
        print("[{}] End grid search".format(get_timestamp()))


if __name__ == '__main__':
    typer.run(main)
