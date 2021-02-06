from celery import Celery, current_app
from dotenv import load_dotenv

def make_celery(app):
    print('=== Connect Celery ===\n\tBroker: {}\n\tBackend: {}'.format(
        app.config['celery']['broker'], app.config['celery']['backend']))
    # Start Celery
    celery = Celery(
        app.config['name'],
        broker= app.config['celery']['broker'],
        backend=app.config['celery']['backend']
    )
    #celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    # Set this as the default celery instance for all threads
    celery.set_default()
    return celery