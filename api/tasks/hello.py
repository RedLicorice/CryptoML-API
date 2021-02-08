import time
from celery import current_app as app

@app.task(name='hello')
def hello(name):
    res = 'Hello ' + name
    time.sleep(5)
    print(res)
    return res