import s3fs
import logging
from cryptoml_core.deps.config import config
from cryptoml_core.util.ident import get_ident

fs: s3fs.S3FileSystem = None
ident: str = None
logging.getLogger('s3fs').setLevel(getattr(logging, config['storage']['s3']['loglevel'].get(str)))
s3_config = {
    'anon':False,
    'use_ssl':config['storage']['s3']['use_ssl'].get(bool),
    'key':config['storage']['s3']['access_key_id'].get(str),
    'secret':config['storage']['s3']['secret_access_key'].get(str),
    'client_kwargs':{
     'endpoint_url': config['storage']['s3']['endpoint'].get(str),
    }
}


def get_fs() -> s3fs.S3FileSystem:
    global fs, ident
    cur_ident = get_ident()
    if not fs or ident != cur_ident:
        fs = s3fs.S3FileSystem(**s3_config)
        ident = cur_ident
    return fs
