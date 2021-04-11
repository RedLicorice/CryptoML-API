from datetime import datetime, timezone, timedelta


def get_timestamp(tz='utc') -> str:
    return datetime.now(getattr(timezone, tz, timezone.utc)).isoformat('T')


def from_timestamp(timestamp: str) -> datetime:
    if timestamp.endswith('Z'):
        timestamp = timestamp[:-1]
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None:
        dt = dt.astimezone(timezone.utc)
    return dt


def to_timestamp(date: datetime) -> str:
    if date.tzinfo is None:
        date = date.astimezone(timezone.utc)
    return date.isoformat('T')


def mul_interval(interval: dict, factor: int):
    seconds = timedelta(**interval).total_seconds()
    return {'seconds': seconds*factor}


def add_interval(timestamp: str, interval: dict):
    date = from_timestamp(timestamp)
    return to_timestamp(date + timedelta(**interval))


def sub_interval(timestamp: str, interval: dict):
    date = from_timestamp(timestamp)
    return to_timestamp(date - timedelta(**interval))


def timestamp_range(start: str, end: str, delta: dict):
    _delta = timedelta(**delta)
    _end = from_timestamp(end)
    curr = from_timestamp(start)
    while curr.timestamp() < _end.timestamp():
        _next = curr + _delta
        if _next.timestamp() > _end.timestamp():
            _next = _end
        yield to_timestamp(curr), to_timestamp(_next)
        curr = _next

def timestamp_windows(start: str, end: str, window: dict, step: dict):
    _step = timedelta(**step)
    _end = from_timestamp(end)
    # Get begin and end of the first window
    _w_begin = from_timestamp(start)
    _w_end = _w_begin + timedelta(**window)

    while _w_end.timestamp() <= _end.timestamp():
        yield to_timestamp(_w_begin), to_timestamp(_w_end)
        # Move window one step forward
        _w_begin += timedelta(**step)
        _w_end += timedelta(**step)
