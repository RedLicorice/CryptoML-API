import pandas as pd
from io import StringIO
from botocore.exceptions import ClientError
import pickle, json
from ..config import config
from ..s3 import get_client
import logging

class StorageService:

    def __init__(self):
        self.s3 = get_client()

    def create_bucket(self, bucket):
        try:
            self.s3.create_bucket(Bucket=bucket)
        except self.s3.exceptions.BucketAlreadyOwnedByYou:
            pass
        except self.s3.exceptions.BucketAlreadyExists:
            pass
        except ClientError as e:
            logging.error(e)

    def exist_file(self, bucket, name):
        response = self.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=name,
        )
        for obj in response.get('Contents', []):
            if obj['Key'] == name:
                return obj['Size']

    def upload_file(self, file, bucket, name):
        self.create_bucket(bucket)
        try:
            self.s3.upload_fileobj(
                file,
                Bucket=bucket,
                Key=name
            )
        except ClientError as e:
            logging.error(e)
            print(e)


    def upload_pickle_obj(self, obj, bucket, name):
        self.create_bucket(bucket)
        try:
            obj = pickle.dumps(obj)
            self.s3.put_object(Body=obj, Bucket=bucket, Key=name, ContentType='application/python-pickle')
            return True
        except TypeError as e:
            logging.error(e)
        except ClientError as e:
            logging.error(e)

    def load_pickled_obj(self, bucket, name):
        try:
            obj_pickled = self.s3.get_object(Bucket=bucket, Key=name)
            obj = pickle.loads(obj_pickled['Body'].read())
            return obj
        except TypeError as e:
            logging.error(e)
        except ClientError as e:
            logging.error(e)
        finally:
            return None

    def upload_json_obj(self, obj, bucket, name):
        self.create_bucket(bucket)
        obj = json.dumps(obj, indent=4, sort_keys=True)
        self.s3.put_object(Body=obj, Bucket=bucket, Key=name, ContentType='application/json')

    def load_json_obj(self, bucket, name):
        obj_pickled = self.s3.get_object(Bucket=bucket, Key=name)
        obj = json.loads(obj_pickled['Body'].read().decode('utf-8'))
        return obj

    def save_df(self, df, bucket, name, **kwargs):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, **kwargs)
        self.create_bucket(bucket)
        self.s3.put_object(Bucket=bucket, Key=name, Body=csv_buffer.getvalue())

    def load_df(self, bucket, name, **kwargs):
        csv_obj = self.s3.get_object(Bucket=bucket, Key=name)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_string), **kwargs)
        return df
