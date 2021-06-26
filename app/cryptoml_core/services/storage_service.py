import pandas as pd
import pickle, json
from cryptoml_core.deps.s3 import get_fs, s3_config
import logging
from shutil import copyfileobj
from functools import lru_cache
from pathlib import Path

global_s3 = get_fs()


@lru_cache
def ls_bucket(bucket):
    return global_s3.ls(bucket)


def exist_file(bucket, name):
    fullpath = bucket + '/' + name
    dirpath = '/'.join(fullpath.split('/')[:-1])
    filelist = ls_bucket(dirpath)
    search_name = f"{bucket}/{name}"
    result = search_name in filelist
    return result


def create_bucket(bucket):
    try:
        global_s3.mkdirs(bucket, exist_ok=True)
        return True
    except Exception as e:
        logging.exception(f'Bucket {bucket} creation failed: {e}')
        logging.exception(e)
        return False


def delete_bucket(bucket):
    try:
        global_s3.delete(bucket, recursive=True)
        return True
    except Exception as e:
        logging.exception(f'Bucket {bucket} delete failed: {e}')
        logging.exception(e)
        return False


def save_file(file, bucket, name):
    try:
        with global_s3.open('{}/{}'.format(bucket, name), 'wb') as dst:
            copyfileobj(file, dst)  # Copy a file object from src to dst
        return name
    except Exception as e:
        logging.exception(e)


def load_file(bucket, name):
    try:
        with global_s3.open('{}/{}'.format(bucket, name), 'rb') as src:
            return src
    except TypeError as e:
        logging.exception(f"TypeError while loading file {name} from bucket {bucket} \n {e}")
        return None
    except FileNotFoundError as e:
        logging.debug(f"File {name} not exist in bucket {bucket}")
        return None
    except Exception as e:
        logging.exception(f"Exception while loading file {name} from bucket {bucket} \n {e}")
        return None


def upload_pickle_obj(obj, bucket, name):
    try:
        with global_s3.open('{}/{}'.format(bucket, name), 'wb') as dst:
            pickle.dump(obj, dst)
        return True
    except TypeError as e:
        logging.exception(f"TypeError while uploading picked object {name} in bucket {bucket} \n {e}")
    except Exception as e:
        logging.exception(f"Exception while uploading picked object {name} in bucket {bucket} \n {e}")


def load_pickled_obj(bucket, name):
    try:
        with global_s3.open('{}/{}'.format(bucket, name), 'rb') as src:
            obj = pickle.load(src)
        return obj
    except TypeError as e:
        logging.exception(f"TypeError while loading picked object {name} from bucket {bucket} \n {e}")
        return None
    except FileNotFoundError as e:
        # logging.error(f"File {name} not exist in bucket {bucket}")
        return None
    except Exception as e:
        logging.exception(f"Exception while loading picked object {name} from bucket {bucket} \n {e}")
        return None


def upload_json_obj(obj, bucket, name):
    create_bucket(bucket)
    with global_s3.open('{}/{}'.format(bucket, name), 'w') as dst:
        json.dump(obj, dst, indent=4, sort_keys=True)


def load_json_obj(bucket, name):
    with global_s3.open('{}/{}'.format(bucket, name), 'r') as src:
        obj = json.load(src)
    return obj


def save_df(df, bucket, name, **kwargs):
    # csv_buffer = StringIO()
    # df.to_csv(csv_buffer, **kwargs)
    # self.create_bucket(bucket)
    # global_s3.put_object(Bucket=bucket, Key=name, Body=csv_buffer.getvalue())
    create_bucket(bucket)
    df.to_csv(
        's3://{}/{}'.format(bucket,name),
        index_label=kwargs.get('index_label', 'time'),
        storage_options=s3_config
    )


def load_df(bucket, name, **kwargs):
    # csv_obj = global_s3.get_object(Bucket=bucket, Key=name)
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # df = pd.read_csv(StringIO(csv_string), **kwargs)
    # return df
    ## Maybe use skipinitialspace=False, / skiprows=None (int) to skip a number of rows, since we already have column names
    return pd.read_csv(
        's3://{}/{}'.format(bucket, name),
        index_col=kwargs.get('index_col', 'time'),
        parse_dates=True,
        storage_options=s3_config
    )