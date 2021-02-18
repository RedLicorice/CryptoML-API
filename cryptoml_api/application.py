from fastapi import FastAPI
from . import endpoints
from cryptoml_common.queue import make_celery
from cryptoml_common.logging import setup_logger

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


