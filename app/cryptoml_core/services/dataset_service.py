from cryptoml_core.repositories.feature_repository import FeatureRepository
from cryptoml_core.models.dataset import Dataset
from cryptoml_core.repositories.dataset_repository import DatasetRepository
from cryptoml_core.util.timestamp import to_timestamp
from cryptoml_core.services.dataset_building_service import DatasetBuildingService
from cryptoml_core.services.storage_service import StorageService
import logging


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
        self.storage: StorageService = StorageService()

    # Use "DatasetBuildingService" to build a dataset from existing bases
    def build_dataset(self, symbol, builder, args):
        service = DatasetBuildingService()
        df = service.build_dataset(symbol, builder, args)
        return self.create_dataset(df, builder, symbol)

    # Import dataset from s3 storage
    def import_from_storage(self, bucket: str, filename: str, name: str, symbol: str, **kwargs):
        df = self.storage.load_df(bucket, filename, parse_dates=True, index_col=kwargs.get('index_col', 'Date'))
        df = df.astype({c: 'float64' for c in df.columns})
        return self.create_dataset(df, name, symbol)

    # Create a Dataset instance
    def create_dataset(self, df, name, symbol):
        storage_path = '{}-{}.csv'.format(name, symbol)
        _first, _last, _indices = get_feature_indices(df)
        self.storage.save_df(df, 'datasets', storage_path)
        self.feature_repo.put_features(df, name, symbol)
        item = Dataset(
            name=name,  # Name of the dataset
            ticker=symbol,  # Ticker name, eg BTC or BTCUSD
            count=df.shape[0],  # Number of entries
            index_min=to_timestamp(df.index.min().to_pydatetime()),  # Timestamp of first record
            index_max=to_timestamp(df.index.max().to_pydatetime()), # Timestamp of last record
            # valid_index_min=to_timestamp(df.first_valid_index()),  # Timestamp of first record
            # valid_index_max=to_timestamp(df.last_valid_index()),  # Timestamp of last record
            valid_index_min=_first,  # Timestamp of first record
            valid_index_max=_last,  # Timestamp of last record
            interval={'days': 1},  # Timedelta args for sampling interval of the features
            features_path=storage_path,  # S3 Storage bucket location
            features=[c for c in df.columns],  # Name of included columns
            feature_indices=_indices
        )
        return self.repo.create(item)


    def fix(self):
        result = []
        for d in self.repo.iterable():
            if not d.feature_indices:
                df = self.feature_repo.get_features(d.name, d.ticker, first=d.index_min, last=d.index_max)
                d.valid_index_min, d.valid_index_max, d.feature_indices = get_feature_indices(df)
                self.repo.update(d.id, d)
                result.append(d)
        return result

    def get_dataset(self, name, symbol) -> Dataset:
        return self.repo.find_by_dataset_and_symbol(name, symbol)

    def find_by_symbol(self, symbol):
        return self.repo.yield_by_symbol(symbol)

    def get_features(self, name, symbol, begin, end, **kwargs):
        # storage_path = '{}-{}.csv'.format(name, symbol)
        # self.storage.load_df('datasets', storage_path)
        return self.feature_repo.get_features(name, symbol, first=begin, last=end, **kwargs)

    def get_target(self, name, symbol, begin, end):
        # storage_path = '{}-{}.csv'.format(name, symbol)
        # self.storage.load_df('datasets', storage_path)
        return self.feature_repo.get_target(name, symbol, first=begin, last=end)
