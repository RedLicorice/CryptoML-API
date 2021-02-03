from flask import request, jsonify
from dependency_injector.wiring import inject, Provide
from ..services import FeaturesService
from ..containers import Container
from ..database import Session

@inject
def index(
        feature_service: FeaturesService = Provide[Container.feature_service],
        session: Session = Provide[Container.db_session]
    ):

    result = {'message':feature_service.hello()}

    return jsonify(result)