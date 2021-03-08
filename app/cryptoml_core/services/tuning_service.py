import pandas as pd
# APP Dependencies
from .storage_service import StorageService
from .dataset_service import DatasetService
from ..exceptions import MessageException
# from ..repositories.classification_repositories import GridSearchRepository
# CryptoML Lib Dependencies
from cryptoml.tuning.grid_search import grid_search
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
from cryptoml_core.deps.dask import get_client
from cryptoml_core.models.classification  import Model, ModelParameters
from cryptoml_core.repositories.classification_repositories import ModelRepository
# from cryptoml_core.models.tuning import GridSearch, ModelTestBlueprint, ModelTest
from cryptoml.util.feature_importances import label_feature_importances
from cryptoml_core.util.dict_hash import dict_hash


class TuningService:
    def __init__(self):
        self.storage = StorageService()
        self.model_repo = ModelRepository()

    # Perform parameter search
    def grid_search(self, model: Model, mp: ModelParameters, **kwargs):
        if not model.id: # Make sure the task exists
            gs = self.model_repo.create(model)
        # Load dataset
        ds = DatasetService()

        X =ds.get_features(model.dataset, model.symbol, mp.cv_interval.begin, mp.cv_interval.end, columns=mp.features)
        y = ds.get_target(model.target, model.symbol, mp.cv_interval.begin, mp.cv_interval.end)

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)

        # Instantiate splitter and scorer
        splitter = BlockingTimeSeriesSplit(n_splits=mp.cv_splits)
        scorer = get_weighted_precision_scorer(weights=mp.precision_weights)

        # Connect to Dask
        dask = get_client()

        # Perform search
        gscv = grid_search(
            est=pipeline_module.estimator,
            parameters=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
            X_train=X,
            y_train=y,
            cv=splitter,
            scoring=scorer,
            sync=kwargs.get('sync', False)
        )

        # Update "Classification" request with found hyperparameters,
        mp.parameter_search_method = 'gridsearch'
        mp.parameters = gscv.best_params_
        tag = "{}-{}-{}-{}-{}"\
            .format(model.symbol, model.dataset, model.target, model.pipeline, dict_hash(mp.parameters))
        mp.result_file = 'cv_results-{}.csv'.format(tag)

        # Parse grid search results to dataframe
        results_df = pd.DataFrame(gscv.cv_results_)

        # Store grid search results on storage
        self.storage.upload_json_obj(mp.parameters, 'grid-search-results', 'parameters-{}.json'\
                                     .format(tag))
        self.storage.save_df(results_df, 'grid-search-results', mp.result_file)

        self.model_repo.append_parameters(model.id, mp)

        return mp


