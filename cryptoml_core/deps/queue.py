from celery import Celery
from cryptoml_core.deps.config import config

def make_celery():
    # Start Celery
    celery = Celery(
        'cryptoml',
        broker= config['celery']['broker'].get(str),
        backend=config['celery']['backend'].get(str)
    )

    # Set this as the default celery instance for all threads
    celery.set_default()

    return celery