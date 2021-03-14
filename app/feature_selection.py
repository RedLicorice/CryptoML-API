import typer
from cryptoml_core.services.tuning_service import TuningService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp


def main(dataset, target, pipeline):
    tuning = TuningService()
    models = ModelService()
    search_models = models.query_models({"dataset": dataset, "target": target, "pipeline": pipeline})
    print("[i] {} models for feature selection".format(len(search_models)))
    for i, m in enumerate(search_models):
        print("==[{}/{}]== MODEL: {} {} {} {} =====".format(i+1, len(search_models), m.symbol, m.dataset, m.target, m.pipeline))
        #mp = tuning.create_parameters_search(m, split=0.7)
        print("[{}] Start grid search".format(get_timestamp()))
        #mp = tuning.grid_search(m, mp, sync="true", verbose=1, n_jobs=8)
        print("[{}] End grid search".format(get_timestamp()))


if __name__ == '__main__':
    typer.run(main)
