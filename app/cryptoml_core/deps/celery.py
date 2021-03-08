from celery import Celery, current_app, states
from cryptoml_core.deps.config import config

def make_celery():
    # Start Celery
    celery = Celery(
        'cryptoml',
        broker=config['database']['redis']['uri'].get(str),
        backend=config['database']['redis']['uri'].get(str)
    )
    celery.conf.update(
        # task_serializer='pickle', # need to setup SSL
        task_track_started=True
    )

    # Set this as the default celery instance for all threads
    celery.set_default()

    return celery
