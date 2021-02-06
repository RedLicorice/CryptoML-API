import pandas as pd
from io import StringIO
from botocore.exceptions import ClientError

class StorageService:

    def __init__(self, client):
        self.s3 = client
        pass

    def upload_fileobj(self, file, bucket, name):
        self.s3.create_bucket(Bucket=bucket)
        self.s3.upload_fileobj(
            file,
            Bucket=bucket,
            Key=name
        )


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
