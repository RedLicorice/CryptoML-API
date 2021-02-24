from cryptoml_core.repositories.feature_repository import FeatureRepository
from cryptoml_core.services.storage_service import StorageService
from cryptoml_core.exceptions import MessageException
from cryptoml.builders import BUILDER_LIST
import importlib, logging, inspect


"""
    Builds base datasets (eg OHLCV, coinmetrics) into useable datasets.
"""
class DatasetBuildingService:
    def __init__(self):
        self.repo: FeatureRepository = FeatureRepository()
        self.storage: StorageService = StorageService()

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
        # if kwargs.get('store', True):
        #     if builder == 'target':
        #         for c in features.columns:
        #             self.repo.put_target(features[c], c, symbol)
        #     else:
        #         self.repo.put_features(features, builder, symbol)
        return features
