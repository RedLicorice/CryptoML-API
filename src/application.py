from flask import Flask, _app_ctx_stack
from .containers import Container
from . import controllers

def create_app() -> Flask:
    container = Container()
    container.config.from_yaml('config.yml')
    container.wire(modules=[controllers])

    app = Flask(__name__)
    app.container = container

    # Register declared routes
    for route in controllers.url_routes:
        app.add_url_rule(**route)

    return app