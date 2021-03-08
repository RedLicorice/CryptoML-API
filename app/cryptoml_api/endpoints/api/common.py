from datetime import datetime
from fastapi import APIRouter

router = APIRouter()


@router.get('/')
def index():
    return "Hello, world!"


@router.get('/healthcheck')
def healthcheck():
    return datetime.now().strftime("%b %d %Y %H:%M:%S")