import pandas as pd
from io import StringIO
import boto3
from botocore.exceptions import ClientError
import pickle, json
from src.config import config

class StorageService:

    def __init__(self):
        _config = config['s3']
        self.s3 = boto3.client(
            service_name='s3',
            endpoint_url=_config['endpoint'].get(str),
            aws_access_key_id=_config['access_key_id'].get(str),
            aws_secret_access_key=_config['secret_access_key'].get(str),
            use_ssl=_config['use_ssl'].get(bool),
            verify=_config['verify'].get(bool)
        )

    def upload_fileobj(self, file, bucket, name):
        self.s3.create_bucket(Bucket=bucket)
        self.s3.upload_fileobj(
            file,
            Bucket=bucket,
            Key=name
        )

    def upload_pickle_obj(self, obj, bucket, name):
        self.s3.create_bucket(Bucket=bucket)
        obj = pickle.dumps(obj)
        self.s3.put_object(Body=obj, Bucket=bucket, Key=name, ContentType='application/python-pickle')

    def load_pickled_obj(self, bucket, name):
        obj_pickled = self.s3.get_object(Bucket=bucket, Key=name)
        obj = pickle.loads(obj_pickled['Body'].read())
        return obj

    def upload_json_obj(self, obj, bucket, name):
        self.s3.create_bucket(Bucket=bucket)
        obj = json.dumps(obj, indent=4, sort_keys=True)
        self.s3.put_object(Body=obj, Bucket=bucket, Key=name, ContentType='application/json')

    def load_json_obj(self, bucket, name):
        obj_pickled = self.s3.get_object(Bucket=bucket, Key=name)
        obj = json.loads(obj_pickled['Body'].read().decode('utf-8'))
        return obj

    def save_df(self, df, bucket, name):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        self.s3.create_bucket(Bucket=bucket)
        self.s3.put_object(Bucket=bucket, Key=name, Body=csv_buffer.getvalue())

    def load_df(self, bucket, name):
        csv_obj = self.s3.get_object(Bucket=bucket, Key=name)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        return df
