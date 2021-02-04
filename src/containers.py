from dependency_injector import containers, providers
import boto3
from . import services
from . import repositories
from .database import get_session_factory


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()
    config.from_yaml('config.yml')

    print("===== CryptoML - API =====")
    print("Providers configuration: {}".format(config.name()))
    print('=== Connect S3 ===\n\tEndpoint: {}\n\tAccess Key: {}'.format(
        config.s3.endpoint(), config.s3.access_key_id()))
    # Connect to AWS S3 Service
    s3_client = providers.Factory(
        boto3.client,
        service_name='s3',
        endpoint_url=config.s3.endpoint(),
        aws_access_key_id=config.s3.access_key_id(),
        aws_secret_access_key=config.s3.secret_access_key(),
        use_ssl=True if not config.s3.endpoint() else config.s3.endpoint().startswith('https'),
        verify=True if not config.s3.endpoint() else config.s3.endpoint().startswith('https')
    )

    dbSessionFactory = providers.Singleton(
        get_session_factory
    )
    feature_repository = providers.Factory(
        repositories.FeatureRepository,
        sessionFactory=dbSessionFactory
    )
    feature_service = providers.Factory(
        services.FeaturesService,
        repository=feature_repository
    )
    storage_service = providers.Factory(
        services.StorageService,
        client=s3_client
    )