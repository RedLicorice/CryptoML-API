from fastapi import APIRouter
from .api import features, common, tuning, tasks, models, datasets

router = APIRouter()


router.include_router(common.router, tags=["common"])
router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
router.include_router(tuning.router, prefix="/tuning", tags=["tuning"])
router.include_router(features.router, prefix="/features", tags=["utils"])
router.include_router(models.router, prefix="/models", tags=["models"])
router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
