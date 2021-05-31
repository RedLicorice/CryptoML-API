from pydantic import BaseModel


# Hold prediction result for a given day
class TimeInterval(BaseModel):
    begin: str
    end: str