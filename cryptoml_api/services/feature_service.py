from ..repositories.feature_repository import FeatureRepository
from .storage_service import StorageService
import pandas as pd
import importlib, logging, inspect
from cryptoml.builders import BUILDER_LIST
from ..exceptions import MessageException
from sklearn.model_selection import train_test_split

class FeatureService:
    def __init__(self):
        self.repo: FeatureRepository = FeatureRepository()
        self.storage: StorageService = StorageService()

    def import_from_storage(self, bucket: str, name: str, dataset: str, symbol: str):
        df = self.storage.load_df(bucket, name, parse_dates=True, index_col='Date')
        self.repo.put_features(df, dataset, symbol)

    def get_builders(self):
        result = {}
        for builder in BUILDER_LIST:
            try:
                builder_module = importlib.import_module('cryptoml.builders.{}'.format(builder))
            except Exception as e:
                continue
            if not hasattr(builder_module, 'build'):
                continue
            params = [p for p in inspect.signature(builder_module.build).parameters if not 'args' in p]
            result[builder] = params
        return result

    def get_builder(self, builder):
        if not builder in BUILDER_LIST:
            raise MessageException('Package cryptoml.builders has no {} module!'.format(builder))
        try:
            builder_module = importlib.import_module('cryptoml.builders.{}'.format(builder))
            if not builder_module:
                raise MessageException('Failed to import cryptoml.builders.{} (importlib returned None)!'.format(builder))
            if not hasattr(builder_module, 'build'):
                raise MessageException('Builder cryptoml.builders.{} has no "build" method!'.format(builder))
        except Exception as e:
            logging.exception(e)
            raise MessageException('Failed to import cryptoml.builders.{} !'.format(builder))
        return builder_module

    def _check_builder_args(self, builder_module, args):
        repl = [p for p in inspect.signature(builder_module.build).parameters if not 'args' in p]
        for p in repl:
            if p not in args.keys():
                raise MessageException('Missing Parameter {} in args!'.format(p))
        return repl

    def check_builder_args(self, builder, args):
        builder_module = self.get_builder(builder)
        return self._check_builder_args(builder_module, args)

    def resolve_builder_args(self, builder_module, args, symbol):
        result = {}
        for p in self._check_builder_args(builder_module, args):
            result[p] = self.repo.get_features(args[p], symbol=symbol)
        return result

    def build_dataset(self, symbol:str, builder: str, args: dict, **kwargs):
        builder_module = self.get_builder(builder)
        args = self.resolve_builder_args(builder_module, args, symbol)
        features = builder_module.build(**args)
        if kwargs.get('store', True):
            if builder == 'target':
                for c in features.columns:
                    self.repo.put_target(features[c], c, symbol)
            else:
                self.repo.put_features(features, builder, symbol)
        return features

    def classification_exists(self, symbol, dataset, target):
        return True

    def get_classification(self, symbol, dataset, target, **kwargs):
        X = self.repo.get_features(dataset, symbol, **kwargs)
        y = self.repo.get_target(target, symbol, **kwargs)
        if X.shape[0] > y.shape[0]:
            X = X.loc[y.index, :]
        else:
            y = y.loc[X.index]
        if kwargs.get('split'):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                shuffle=False,
                train_size=kwargs.get('split')
            )
            return X_train, X_test, y_train, y_test
        return X, None, y, None
