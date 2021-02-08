from fastapi import APIRouter
from .api import features, common, model_training



router = APIRouter()
router.include_router(common.router, tags=["common"])
router.include_router(model_training.router, prefix="/ml", tags=["utils"])
router.include_router(features.router, prefix="/features", tags=["utils"])