from dependency_injector import containers, providers
from . import services
from .database import get_session_factory

class Container(containers.DeclarativeContainer):

    config = providers.Configuration()
    dbSession = get_session_factory()

    db_session = providers.Callable(
        dbSession
    )
    feature_service = providers.Factory(
        services.FeaturesService,
        db_session=db_session
    )