from .src.models import grid_search, train_model, test_model, predict_model
import dask.dataframe as dd
import importlib

# Launch a Distributed grid search on the cluster
def launch_grid_search(pipeline: str, bucket: str, features_path: str, target_path: str):
    p = importlib.import_module('cryptoml.pipelines.' + pipeline)
    features = dd.read_csv('s3://{}/{}.csv'.format(bucket, features_path))
    target =  dd.read_csv('s3://{}/{}.csv'.format(bucket, features_path))
