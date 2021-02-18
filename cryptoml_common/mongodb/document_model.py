from pydantic import BaseModel, Field, Extra, root_validator
from typing import Optional

class DocumentModel(BaseModel):
    id: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None

    def dict(self, include_nulls=False, **kwargs):
        """Override the super dict method by removing null keys from the dict, unless include_nulls=True"""
        kwargs["exclude_none"] = not include_nulls
        return super().dict(**kwargs)

    @root_validator(pre=True)
    def _min_properties(cls, data):
        """At least one property is required"""
        if not data:
            raise ValueError("At least one property is required")
        return data

    @root_validator(pre=True)
    def _set_id(cls, data):
        """sync data id field to mongodb's _id field"""
        document_id = data.get("_id")
        if document_id:
            data["id"] = document_id
        return data

    class Config:
        extra = Extra.ignore  # forbid sending additional fields/properties
        anystr_strip_whitespace = True  # strip whitespaces from strings