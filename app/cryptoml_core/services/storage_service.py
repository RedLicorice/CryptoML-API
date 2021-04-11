import pandas as pd
import dask.dataframe as dd
import pickle, json
from cryptoml_core.deps.s3 import get_fs, s3_config
import logging
from shutil import copyfileobj


"""
    Handles object storage to and from S3
"""
class StorageService:

    def __init__(self):
        self.s3 = get_fs()

    def create_bucket(self, bucket):
        try:
            self.s3.mkdirs(bucket, exist_ok=True)
        except Exception as e:
            print('bucket creation failed: {}'.format(bucket))
            logging.exception(e)

    def exist_file(self, bucket, name):
        return name in self.s3.ls(bucket)

    def upload_file(self, file, bucket, name):
        self.create_bucket(bucket)
        try:
            with self.s3.open('{}/{}'.format(bucket, name), 'wb') as dst:
                copyfileobj(file, dst) # Copy a file object from src to dst
        except Exception as e:
            logging.exception(e)

    def upload_pickle_obj(self, obj, bucket, name):
        self.create_bucket(bucket)
        try:
            with self.s3.open('{}/{}'.format(bucket, name), 'wb') as dst:
                pickle.dump(obj, dst)
            return True
        except TypeError as e:
            logging.exception(e)
        except Exception as e:
            logging.exception(e)

    def load_pickled_obj(self, bucket, name):
        try:
            with self.s3.open('{}/{}'.format(bucket, name), 'rb') as src:
                obj = pickle.load(src)
            return obj
        except TypeError as e:
            logging.exception(e)
        except Exception as e:
            logging.exception(e)
        finally:
            return None

    def upload_json_obj(self, obj, bucket, name):
        self.create_bucket(bucket)
        with self.s3.open('{}/{}'.format(bucket, name), 'w') as dst:
            json.dump(obj, dst, indent=4, sort_keys=True)

    def load_json_obj(self, bucket, name):
        with self.s3.open('{}/{}'.format(bucket, name), 'r') as src:
            obj = json.load(src)
        return obj

    def save_df(self, df, bucket, name, **kwargs):
        # csv_buffer = StringIO()
        # df.to_csv(csv_buffer, **kwargs)
        # self.create_bucket(bucket)
        # self.s3.put_object(Bucket=bucket, Key=name, Body=csv_buffer.getvalue())
        self.create_bucket(bucket)
        df.to_csv(
            's3://{}/{}'.format(bucket,name),
            index_label=kwargs.get('index_label', 'time'),
            storage_options=s3_config
        )

    def load_df(self, bucket, name, **kwargs):
        # csv_obj = self.s3.get_object(Bucket=bucket, Key=name)
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

    def load_dask_df(self, bucket, name, **kwargs):
        return dd.read_csv(
            's3://{}/{}'.format(bucket,name),
            index_col=kwargs.get('index_col', 'time'),
            parse_dates=True,
            storage_options=s3_config
        )
