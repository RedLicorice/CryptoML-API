from celery import current_app
import celery.states as states
from fastapi import APIRouter, Depends, Body, HTTPException
from ...services.model_service import ModelService, TrainingParameters

router = APIRouter()


@router.post('/train')
def train_model(training: TrainingParameters = Body(...)):
    pass

@router.post('/test')
def test_model(training: TrainingParameters = Body(...)):
    pass

@router.post('/predict')
def predict_model():
    pass
