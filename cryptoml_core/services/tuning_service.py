import pandas as pd
# APP Dependencies
from .storage_service import StorageService
from .dataset_service import DatasetService
from ..exceptions import MessageException
from ..repositories.classification_repositories import GridSearchRepository
# CryptoML Lib Dependencies
from cryptoml.models.grid_search import grid_search
from cryptoml.util.weighted_precision_score import get_weighted_precision_scorer
from cryptoml.util.blocking_timeseries_split import BlockingTimeSeriesSplit
from cryptoml.pipelines import get_pipeline
# CryptoML Common Dependencies
from cryptoml_core.models.tuning import GridSearch, ModelTestBlueprint, ModelTest
from cryptoml.util.feature_importances import label_feature_importances
from typing import List

def get_gs_name(gs: GridSearch):
    return "{}-{}-{}-{}.csv".format(gs.dataset, gs.target, gs.symbol, gs.id)

class TuningService:
    def __init__(self):
        self.storage = StorageService()
        self.gs_repo = GridSearchRepository()

    # Perform parameter search
    def grid_search(self, gs: GridSearch, **kwargs):
        if not gs.id: # Make sure the task exists
            gs = self.gs_repo.create(gs)
        # Load dataset
        ds = DatasetService()
        X =ds.get_features(gs.dataset, gs.symbol, gs.cv_begin, gs.cv_end)
        y = ds.get_target(gs.target, gs.symbol, gs.cv_begin, gs.cv_end)

        # Load pipeline
        pipeline_module = get_pipeline(gs.pipeline)

        # Instantiate splitter and scorer
        splitter = BlockingTimeSeriesSplit(n_splits=gs.cv_splits)
        scorer = get_weighted_precision_scorer(weights=gs.precision_weights)

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
        gs.parameters = gscv.best_params_
        gs.result_file = get_gs_name(gs)
        gs.feature_importances = label_feature_importances(gscv.best_estimator_, X.columns)
        self.gs_repo.update(gs.id, gs)

        # Parse grid search results to dataframe
        results_df = pd.DataFrame(gscv.cv_results_)

        # Store grid search results on storage
        self.storage.upload_json_obj(gs.parameters, 'grid-search-results', 'parameters-{}.json'\
                                     .format(gs.result_file))
        self.storage.save_df(results_df, 'grid-search-results', 'cv_results-{}.csv'\
                             .format(gs.result_file))

        return gs

    def tests_from_blueprint(self, gs: GridSearch, bp: ModelTestBlueprint):
        if not gs.parameters:
            raise MessageException("GridSearch has no results!")
        for w in bp.windows:
            yield ModelTest(
                symbol=gs.symbol,
                dataset=gs.dataset,
                target=gs.target,
                # What pipeline should be used
                pipeline=gs.pipeline,
                parameters=gs.parameters,
                features=gs.features,
                window=w,
                test_begin=bp.test_begin,
                test_end=bp.test_end
            )




