from dask.distributed import Client
from cryptoml_core.deps.config import config
from cryptoml_core.util.ident import get_ident

dask: Client = None
ident: str = None

def get_client() -> Client:
    global dask, ident
    cur_ident = get_ident()
    if not dask or cur_ident != ident:
        dask = Client(config['dask']['scheduler'].get(str))
        ident = cur_ident
    return dask
