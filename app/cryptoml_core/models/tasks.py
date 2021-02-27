from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel
from typing import Optional
from celery import states


class Task(DocumentModel):
    task_id: str
    name: Optional[str] = None
    status: Optional[str] = states.PENDING
