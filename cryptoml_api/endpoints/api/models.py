from fastapi import APIRouter, Body
from cryptoml_core.models.classification import SlidingWindowClassification
from cryptoml_core.deps.celery import current_app, states

router = APIRouter()


@router.post('/train')
def train_model(training: SlidingWindowClassification = Body(...)):
    pass

@router.post('/test')
def test_model(training: SlidingWindowClassification = Body(...)):
    pass

@router.post('/predict')
def predict_model():
    pass
