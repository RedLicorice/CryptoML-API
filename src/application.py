from fastapi import FastAPI
from .queue import make_celery
from . import endpoints
from . import tasks
from .config import config

app = None
celery = None

def create_app(routes=True) -> FastAPI:
    global app, celery

    # Start Celery
    celery = make_celery()

    # Start FastAPI
    app = FastAPI()
    app.include_router(endpoints.router)
    app.celery = celery

    return app


