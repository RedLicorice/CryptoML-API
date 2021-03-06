from fastapi import FastAPI
from . import endpoints
from cryptoml_core.deps.celery import make_celery
from cryptoml_core.logging import setup_logger
from .middleware import request_handler

app = None
celery = None

def create_worker():
    global celery
    # Setup Logging
    setup_logger()

    # Start Celery
    celery = make_celery()

    return celery


def create_app() -> FastAPI:
    global app, celery
    # Setup Logging
    setup_logger()

    # Start Celery
    celery = make_celery()

    # Start FastAPI
    app = FastAPI()
    app.middleware("http")(request_handler)
    app.include_router(endpoints.router)
    app.celery = celery

    return app

if __name__ == '__main__':
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)