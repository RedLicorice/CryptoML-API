from cryptoml_core.deps.mongodb.document_repository import DocumentNotFoundException
from cryptoml_core.exceptions import MessageException
from cryptoml_core.repositories.feature_repository import FeatureRepository
from cryptoml_core.models.dataset import Dataset
from cryptoml_core.repositories.dataset_repository import DatasetRepository, FeatureSelection
from cryptoml_core.util.timestamp import to_timestamp, from_timestamp, add_interval, mul_interval
from cryptoml_core.services.dataset_building_service import DatasetBuildingService
import cryptoml_core.services.storage_service as storage_service
import logging
import math
from typing import List, Optional
import pandas as pd


def get_feature_indices(df):
    feature_indices = {}
    global_first = None
    global_last = None
    for c in df.columns:
        # If all elements are non-NA/null, first_valid_index and last_valid_index return None.
        # They also return None for empty Series/DataFrame.
        if df[c].empty:
            logging.exception("Feature {} is empty!".format(c))
            continue
        fvi = df[c].first_valid_index() or df[c].index.min()
        lvi = df[c].last_valid_index() or df[c].index.max()
        _first = fvi.to_pydatetime()
        _last = lvi.to_pydatetime()
        feature_indices[c] = {'first': to_timestamp(_first), 'last': to_timestamp(_last)}
        if not global_first or _first > global_first:
            global_first = _first
        if not global_last or _last < global_last:
            global_last = _last
    return to_timestamp(global_first), to_timestamp(global_last), feature_indices


class DatasetService:
    def __init__(self):
        self.repo: DatasetRepository = DatasetRepository()
        self.feature_repo: FeatureRepository = FeatureRepository()
        
    # Use "DatasetBuildingService" to build a dataset from existing bases
    def build_dataset(self, symbol, builder, args):
        service = DatasetBuildingService()
        df = service.build_dataset(symbol, builder, args)
        return self.create_dataset(df, builder, symbol)

    # Import dataset from s3 storage
    def import_from_storage(self, bucket: str, filename: str, name: str, symbol: str, **kwargs):
        df = storage_service.load_df(bucket, filename, parse_dates=True, index_col=kwargs.get('index_col', 'Date'))
        df = df.astype({c: 'float64' for c in df.columns})
        return self.create_dataset(df, name, symbol)

    # Create a Dataset instance
    def create_dataset(self, df, name, symbol, type='FEATURES'):
        storage_path = '{}-{}.csv'.format(name, symbol)
        _first, _last, _indices = get_feature_indices(df)

        storage_service.save_df(df, 'datasets', storage_path)
        self.feature_repo.delete_features(dataset=name, symbol=symbol)
        self.feature_repo.put_features(df, name, symbol)
        item = Dataset(
            name=name,  # Name of the dataset
            symbol=symbol,  # Ticker name, eg BTC or BTCUSD
            type=type,
            count=df.shape[0],  # Number of entries
            index_min=to_timestamp(df.index.min().to_pydatetime()),  # Timestamp of first record
            index_max=to_timestamp(df.index.max().to_pydatetime()),  # Timestamp of last record
            valid_index_min=_first,  # Timestamp of first record
            valid_index_max=_last,  # Timestamp of last record
            interval={'days': 1},  # Timedelta args for sampling interval of the features
            features_path=storage_path,  # S3 Storage bucket location
            features=[c for c in df.columns],  # Name of included columns
            feature_indices=_indices
        )
        try:
            existing_ds = self.repo.find_by_dataset_and_symbol(dataset=item.name, symbol=item.symbol)
            logging.info(f"Delete old dataset {item.name} {item.symbol} CRE: {existing_ds.created} UPD: {existing_ds.updated}")
            self.repo.delete(existing_ds.id)
        except DocumentNotFoundException as e:
            pass

        new_ds = self.repo.create(item)
        logging.info(f"Create dataset {item.name} {item.symbol} ID: {new_ds.id}")
        return new_ds

    def merge_datasets(self, datasets: List[Dataset], name: str, symbol: str):
        dfs = [self.get_dataset_features(d) for d in datasets]
        columns = []
        for df in dfs:
            drop = [c for c in df.columns if c in columns]
            if drop:
                df.drop(drop, axis='columns', inplace=True)
            columns = columns + [c for c in df.columns]

        df = pd.concat(dfs,
                       axis='columns',
                       # verify_integrity=True,
                       sort=True,
                       join='inner'
                       )
        ds = self.create_dataset(df, name, symbol)
        return ds

    def get(self, id) -> Dataset:
        return self.repo.get(id)

    def get_dataset(self, name, symbol) -> Dataset:
        return self.repo.find_by_dataset_and_symbol(name, symbol)

    def find_by_symbol(self, symbol):
        return self.repo.yield_by_symbol(symbol)

    def get_dataset_symbols(self, name: str):
        return self.repo.get_symbols(name)

    def get_features(self, name, symbol, begin, end, **kwargs):
        # storage_path = '{}-{}.csv'.format(name, symbol)
        # storage_service.load_df('datasets', storage_path)
        return self.feature_repo.get_features(name, symbol, first=begin, last=end, **kwargs)

    def get_dataset_features(self, ds: Dataset, **kwargs):  # begin and end
        # storage_path = '{}-{}.csv'.format(name, symbol)
        # storage_service.load_df('datasets', storage_path)
        begin = kwargs.get('begin', ds.index_min)
        end = kwargs.get('end', ds.index_max)
        if 'begin' in kwargs:
            del kwargs['begin']
        if 'end' in kwargs:
            del kwargs['end']
        if 'method' in kwargs and 'target' in kwargs:
            fs = DatasetService.get_feature_selection(ds, kwargs.get('method'), kwargs.get('target'))
            if not fs:
                raise MessageException(f"Failed to find feature selection {kwargs.get('method')} for dataset {ds.name}.{ds.symbol}")
            return self.get_features(name=ds.name, symbol=ds.symbol, begin=begin, end=end, columns=fs.features)
        return self.get_features(name=ds.name, symbol=ds.symbol, begin=begin, end=end, columns=kwargs.get('columns'))

    def get_target(self, name, symbol, begin, end):
        # storage_path = '{}-{}.csv'.format(name, symbol)
        # storage_service.load_df('datasets', storage_path)
        return self.feature_repo.get_target(name, symbol, first=begin, last=end)

    def get_dataset_target(self, ds: Dataset, name: str, **kwargs):
        begin = kwargs.get('begin', ds.index_min)
        end = kwargs.get('end', ds.index_max)
        if 'begin' in kwargs:
            del kwargs['begin']
        if 'end' in kwargs:
            del kwargs['end']
        return self.get_target(
            name=name,
            symbol=ds.symbol,
            begin=begin,
            end=end
        )

    def all(self):
        return self.repo.iterable()

    def by_type(self, type: str):
        return [d for d in self.repo.yield_by_type(type)]

    def query(self, query, projection: Optional[dict] = None):
        return self.repo.query(query, projection)

    @staticmethod
    def get_train_test_split_indices(dataset: Dataset, split: float):
        if split > 1.0:
            split = 1.0
        if split < 0.0:
            split = 0.0

        search_points = math.floor(dataset.count * split)
        search_end = add_interval(dataset.valid_index_min, mul_interval(dataset.interval, search_points))
        return {
            "train": {"begin": dataset.valid_index_min, "end": search_end},
            "test": {"begin": search_end, "end": dataset.valid_index_max}
        }

    def append_feature_selection(self, ds: Dataset, fs: FeatureSelection):
        ds.feature_selection.append(fs)
        return self.repo.update(ds.id, ds)

    @staticmethod
    def has_feature_selection(ds: Dataset, method: str, target: str):
        for fs in ds.feature_selection:
            if fs.method == method and fs.target == target:
                return True
        return False

    @staticmethod
    def get_feature_selection(ds: Dataset, method: str, target: str) -> FeatureSelection:
        for i in range(len(ds.feature_selection)-1, -1, -1):
            if ds.feature_selection[i].method == method and ds.feature_selection[i].target == target:
                return ds.feature_selection[i]
        return None

    def remove_feature_selection(self, ds: Dataset, method: str, target: str):
        found = None
        for i in range(len(ds.feature_selection)):
            if ds.feature_selection[i].method == method and ds.feature_selection[i].target == target:
                found = i
        if found is not None:
            del ds.feature_selection[found]
            self.repo.update(ds.id, ds)
            return True
        return False

    def get_x_y(self, dataset, symbol, target, features, begin, end):
        # Load features from the dataset, using indicated feature selection method
        ds = self.get_dataset(name=dataset, symbol=symbol)
        fs = DatasetService.get_feature_selection(ds=ds, method=features, target=target)
        if not fs:
            logging.warning(f"Feature selection with method {features} does not exist in dataset"
                            f" {dataset}.{symbol}")

        X_train = self.get_dataset_features(ds=ds, begin=begin, end=end, columns=fs.features)
        y_train = self.get_target(target, symbol, begin=begin, end=end)

        return X_train, y_train