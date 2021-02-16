from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta


class TrainingParameters(BaseModel):
    day: Optional[datetime] = datetime.now(timezone.utc)
    window: Optional[int] = 30
    window_interval: Optional[str] = 'days'
    symbol: str
    dataset: str
    target: str
    pipeline: str
    parameters: Optional[dict] = {}


class TrainingParametersUtils:
    @staticmethod
    def train_begin(model: TrainingParameters):
        delta = timedelta(**{model.window_interval: model.window})
        begin = model.day - delta
        # Datetime should be in UTC, so throw a "Z" (for Zulu timezone) on the end to mark the "timezone" as UTC
        return begin.isoformat("T") + "Z"

    @staticmethod
    def train_end(model: TrainingParameters):
        return model.day.isoformat("T") + "Z"

    @staticmethod
    def test_begin(model: TrainingParameters):
        return model.day.isoformat("T") + "Z"

    @staticmethod
    def test_end(model: TrainingParameters):
        delta = timedelta(**{model.window_interval: 1})
        end = model.day + delta
        return end.isoformat("T") + "Z"

    @staticmethod
    def day_str(model: TrainingParameters):
        return model.day.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_tag(model: TrainingParameters):
        return '{}-{}-{}_W{}-D{}T{}'.format(
            model.pipeline,
            model.symbol,
            model.day_str(),
            model.window,
            model.dataset,
            model.target
        )
