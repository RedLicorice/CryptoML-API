from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel
from typing import Optional
from celery import states


class Task(DocumentModel):
    name: Optional[str] = None
    task_name: Optional[str] = None
    batch: Optional[str] = None
    status: Optional[str] = states.PENDING
    args: Optional[dict] = None
    result: Optional[dict] = None
    completed_at: Optional[str] = None
