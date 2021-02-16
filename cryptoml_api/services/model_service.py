import importlib
import pandas as pd
from .storage_service import StorageService
from ..exceptions import MessageException
from ..repositories.model_repository import ModelRepository, TrainingParameters, PersistedModel
from .feature_service import FeatureService
from cryptoml.pipelines import PIPELINE_LIST
from cryptoml.models.grid_search import grid_search
from cryptoml.models.testing import trailing_window_test
from cryptoml.models.training import train_model

from sklearn.metrics import classification_report
import logging


class ModelService:
    def __init__(self):
        self.storage = StorageService()

    def get_pipeline(self, pipeline):
        if not pipeline in PIPELINE_LIST:
            raise MessageException('Package cryptoml.pipelines has no {} module!'.format(pipeline))
        try:
            pipeline_module = importlib.import_module('cryptoml.pipelines.{}'.format(pipeline))
            if not pipeline_module:
                raise MessageException(
                    'Failed to import cryptoml.pipelines.{} (importlib returned None)!'.format(pipeline))
            if not hasattr(pipeline_module, 'estimator'):
                raise MessageException('Builder cryptoml.pipelines.{} has no "estimator" attribute!'.format(pipeline))
        except Exception as e:
            logging.exception(e)
            raise MessageException('Failed to import cryptoml.pipelines.{} !'.format(pipeline))
        return pipeline_module

    def grid_search(self, pipeline: str, features: pd.DataFrame, target: pd.Series, **kwargs):
        pipeline_module = self.get_pipeline(pipeline)
        gscv = grid_search(
            est=pipeline_module.estimator,
            parameters=kwargs.get('parameter_grid', pipeline_module.PARAMETER_GRID),
            X_train=features.values,
            y_train=target.values,
            **kwargs
        )
        results_df = pd.DataFrame(gscv.cv_results_)
        return results_df, gscv.best_params_

    def test_model(self,
                   pipeline: str,
                   parameters: dict,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   **kwargs
                ):
        try:
            pipeline_module = self.get_pipeline(pipeline)
            labels, predictions = trailing_window_test(
                est=pipeline_module.estimator,
                parameters=parameters,
                W=kwargs.get('W', 30),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
            clf_report = classification_report(labels=labels, predictions=predictions)
            report = pd.DataFrame.from_dict(clf_report, orient='index')
            return report
        except Exception as e:
            logging.exception(e)
            return None

    def train_model_day(self, training: TrainingParameters, **kwargs) -> PersistedModel:
        pipeline_module = self.get_pipeline(training.pipeline)
        fs = FeatureService()
        # Get training window datasets
        X_train, _, y_train, _ = fs.get_classification(
            symbol=training.symbol,
            dataset=training.dataset,
            target=training.target,
            begin=training.train_begin(),
            end=training.train_end()
        )
        # If kwargs specifies a subset of features, only use those for training
        feature_subset = kwargs.get('train_features', [c for c in X_train.columns])
        X_train = X_train.loc[:, feature_subset]
        # Train a model using the specified parameters
        _est = train_model(
            est=pipeline_module.estimator,
            parameters=training.parameters,
            X_train=X_train,
            y_train=y_train
        )
        # Create a persisted model instance
        model = PersistedModel(
            estimator=_est,
            features=feature_subset,
            training=training
        )
        return model
