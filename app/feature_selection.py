import typer
from cryptoml_core.services.tuning_service import TuningService
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp


def main(dataset: str, target: str, pipeline: str):
    tuning = TuningService()
    models = ModelService()
    query = {"dataset": dataset, "target": target, "pipeline": pipeline}
    if pipeline == 'all':
        del query['pipeline']
    if target == 'all':
        del query['target']
    models.clear_features(query)
    search_models = models.query_models(query)
    print("[i] {} models for feature selection".format(len(search_models)))
    for i, m in enumerate(search_models):
        print("==[{}/{}]== MODEL: {} {} {} {} =====".format(i+1, len(search_models), m.symbol, m.dataset, m.target, m.pipeline))
        mf = tuning.create_features_search(model=m, split=0.7, method='importances')
        print("[{}] Start feature search".format(get_timestamp()))
        mf = tuning.feature_selection(m, mf, sync=True)
        print("[{}] End feature search".format(get_timestamp()))


if __name__ == '__main__':
    typer.run(main)
