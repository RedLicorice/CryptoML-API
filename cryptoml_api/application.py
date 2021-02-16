from fastapi import FastAPI
from .queue import make_celery
from . import endpoints
from .config import config
from .logging import setup_logger
from .dask import make_dask_client

app = None
celery = None

def create_worker():
    global celery, dask
    # Setup Logging
    setup_logger()

    # Start Celery
    celery = make_celery()

    # Start Dask Client
    # dask = make_dask_client()

    return celery


def create_app() -> FastAPI:
    global app, celery, dask
    # Setup Logging
    setup_logger()

    # Start Celery
    celery = make_celery()

    # Start Dask Client
    # dask = make_dask_client()

    # Start FastAPI
    app = FastAPI()
    app.include_router(endpoints.router)
    app.celery = celery

    return app


