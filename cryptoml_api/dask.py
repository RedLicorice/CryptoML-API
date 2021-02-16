from .config import config
from dask.distributed import Client, LocalCluster

def make_dask_client():
    return Client(config['dask']['scheduler'].get(str))

def make_dask_local_client():
    cluster = LocalCluster(n_workers=2, dashboard_address='localhost:8788')
    return Client(cluster)

