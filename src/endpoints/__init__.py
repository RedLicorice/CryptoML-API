from fastapi import APIRouter
from .api import features, common, tasks



router = APIRouter()
router.include_router(common.router, tags=["common"])
router.include_router(tasks.router, prefix="/tasks", tags=["utils"])
router.include_router(features.router, prefix="/features", tags=["utils"])