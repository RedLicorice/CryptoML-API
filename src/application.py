from flask import Flask, _app_ctx_stack
from flask_cli import FlaskCLI
from .containers import Container
from .queue import make_celery
from dotenv import load_dotenv

# Initializes Container, Flask and Celery
# ie. the application's base, without routes or tasks
def create_base() -> Flask:
    # Load .env so variables can be interpolated in config.yml
    load_dotenv()
    # Initialize services container
    container = Container()
    
    # Start Flask
    app = Flask(__name__)
    app.container = container
    app.config.update(container.config())

    # Start FlaskCLI
    FlaskCLI(app)
    
    # Start Celery
    celery = make_celery(app)
    app.celery = celery

    return app

# Wires up Flask controllers on base app, returns Flask instance
def create_app() -> Flask:
    app = create_base()

    # Import app controllers so they are only available for main app
    # and wire them to the app's container
    from . import controllers
    app.container.wire(modules=[controllers])

    # Register declared routes in app
    for route in controllers.url_routes:
        app.add_url_rule(**route)

    return app


# Wires up Celery tasks on base app, returns Celery Instance
def create_worker():
    app = create_base()
    # Import app tasks so they are only available for worker app
    # and wire them to the app's container
    from . import tasks
    app.container.wire(modules=[tasks])

    return app.celery


