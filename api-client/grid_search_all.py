import requests
import typer
import itertools
import math
import random
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional


grid_search_url = 'http://127.0.0.1:8000/tuning/gridsearch-batch'
datasets_url = 'http://127.0.0.1:8000/datasets'
task_status_url = 'http://127.0.0.1:8000/tasks/status'

PIPELINE_LIST = [
    'adaboost_decisiontree',
    'bagging_decisiontree',
    'bagging_linear_svc',
    'bagging_poly_svc',
    'bagging_rbf_svc',
    #'debug_xgboost',
    'plain_knn',
    'plain_linear_svc',
    'plain_mlp',
    'plain_mnb',
    'plain_poly_svc',
    'plain_randomforest',
    'plain_rbf_svc',
    'plain_xgboost'
]
TARGET_LIST = [
    'class',
    'binary',
    'bin_class',
    'bin_binary'
]
SKIP_DATASET = [
    'ohlcv',
    'coinmetrics',
    'kraken_ohlcv',
    'target'
]


def from_timestamp(timestamp: str):
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


def get_datasets():
    res = requests.get(datasets_url)
    res = res.json()
    return [d for d in res if d['name'] != 'target'], [d for d in res if d['name'] == 'target']


def get_request(pipeline, target, d):
    first = from_timestamp(d['valid_index_min'])
    last = from_timestamp(d['valid_index_max'])
    # Get a timedelta object, convert it to number of seconds. Divide the value by seconds in a day (86400) to get
    # span of valid data (in days)
    valid_days = math.floor((last - first).total_seconds() / 86400)
    # Perform grid search on 70% of valid days
    search_days = math.floor(valid_days * 0.7)
    search_end = first + timedelta(days=search_days)

    return {
        "task": {
            "symbol": d['ticker'],
            "dataset": d['name'],
            "target": target,
            "pipeline": pipeline,
            "cv_begin": to_timestamp(first),
            "cv_end": to_timestamp(search_end),
            "cv_splits": 5,
            "precision_weights": {"0": 1.0, "1": 0.8, "2": 1.0}
        },
        "tests": {
            "windows": [
                {"days": 30},
                {"days": 90},
                {"days": 150}
            ],
            "test_begin": to_timestamp(search_end),
            "test_end": to_timestamp(last)
        }
    }


def create(dirname: str):
    datasets, targets = get_datasets()
    # batch by pipeline
    res = {}
    for pipeline, target, d in itertools.product(PIPELINE_LIST, TARGET_LIST, datasets):
        if d['name'] in SKIP_DATASET:
            continue
        tag = "{}-{}-{}".format(d['ticker'], d['name'], target)
        if not tag in res:
            res[tag] = []
        res[tag].append(get_request(pipeline, target, d))
    os.makedirs(dirname, exist_ok=True)
    for name, items in res.items():
        with open('{}/{}.json'.format(dirname, name)) as f:
            json.dump(items, f)


if __name__ == "__main__":
    random.seed(datetime.utcnow().timestamp())
    typer.run(create)