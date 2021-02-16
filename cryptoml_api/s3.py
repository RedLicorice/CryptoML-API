import boto3
from botocore.client import BaseClient
from .config import config


client: BaseClient = None

def get_client() -> BaseClient:
    global client
    if not client:
        client = boto3.client(
            service_name='s3',
            endpoint_url=config['storage']['s3']['endpoint'].get(str),
            aws_access_key_id=config['storage']['s3']['access_key_id'].get(str),
            aws_secret_access_key=config['storage']['s3']['secret_access_key'].get(str),
            use_ssl=config['storage']['s3']['use_ssl'].get(bool),
            verify=config['storage']['s3']['verify'].get(bool)
        )
    return client
