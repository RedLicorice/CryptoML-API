from fastapi import APIRouter
from .api import features, common, gridsearch



router = APIRouter()
router.include_router(common.router, tags=["common"])
router.include_router(gridsearch.router, prefix="/ml", tags=["utils"])
router.include_router(features.router, prefix="/features", tags=["utils"])