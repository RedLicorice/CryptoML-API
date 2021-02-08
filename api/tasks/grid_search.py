import time
from celery import current_app as app
from ..services import FeaturesService, StorageService

@app.task(name='gridsearch')
def grid_search(name):
    featuresService: FeaturesService = FeaturesService()
    storage: StorageService = StorageService()
    print("FS: {} Storage: {}".format(featuresService, storage))
    res = featuresService.hello() + name
    time.sleep(5)
    print(res)