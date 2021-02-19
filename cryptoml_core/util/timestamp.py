from datetime import datetime, timezone, timedelta


def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat('T')+'Z'


def parse_timestamp(timestamp: str):
    if timestamp.endswith('Z'):
        timestamp = timestamp[:-1]
    return datetime.fromisoformat(timestamp)


def to_timestamp(date: datetime) -> str:
    return date.isoformat('T')+'Z'


def add_interval(timestamp: str, amount: int, interval: str = 'days'):
    date = parse_timestamp(timestamp)
    return to_timestamp(date + timedelta(**{interval: amount}))

def sub_interval(timestamp: str, amount: int, interval: str = 'days'):
    date = parse_timestamp(timestamp)
    return to_timestamp(date - timedelta(**{interval: amount}))