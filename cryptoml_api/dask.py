from dask.distributed import Client
from .config import config

dask_client = Client(config['dask']['scheduler'].get(str))