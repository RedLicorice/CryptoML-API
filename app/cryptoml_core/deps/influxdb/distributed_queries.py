from cryptoml_core.util.timestamp import timestamp_range
from cryptoml_core.deps.influxdb.client import DataFrameClient
from cryptoml_core.deps.influxdb.queries import query_first_timestamp, query_last_timestamp, query_meta, append_tags
from joblib import delayed
import pandas as pd

__all__ = ("query_dask_dataframe")


# Dask.delayed executor for fetching part of a dataframe
@delayed
def query_interval(measure, begin, end, tags=None, meta=None) -> pd.DataFrame:
    with DataFrameClient() as client:
        query = "SELECT {} FROM {} WHERE".format(','.join([c for c in meta.columns]), measure)
        query += " time >= '{}' AND time < '{}'".format(begin, end)
        query = append_tags(query, tags)
        query += " ORDER BY time"
        # if tags:
        #     for tag, value in tags.items():
        #         if not value:
        #             continue
        #         query += " AND {}='{}'".format(tag, value)
        res = client.query(query)
        data = res[measure]
        return meta.append(data)

def query_range(measure, ranges, **kwargs):
    for b,e in ranges:
        yield query_interval(measure=measure, begin=b, end=e, **kwargs)


def query_delayed_dataframe(measure, first=None, last=None, **kwargs)-> pd.DataFrame:
    tags = kwargs.get('tags')
    cols = kwargs.get('columns')
    if not first:
        first = query_first_timestamp(measure=measure, tags=tags)
    if not last:
        last = query_last_timestamp(measure=measure, tags=tags)
    delta = kwargs.get('delta', {'days': 365}) # fetch one year at a time
    ranges = timestamp_range(start=first, end=last, delta=delta)
    meta = query_meta(measure=measure, columns=cols)
    # ddf = pd.from_delayed(
    #     dfs=query_range(measure, ranges, tags=tags, meta=meta),
    #     meta=meta,
    #     # divisions=tuple(b for b, e in ranges)
    # )
    dfs = [query_interval(measure=measure, begin=b, end=e, columns=[c for c in meta.columns]) for b, e in ranges]
    ddf = pd.concat([meta]+dfs, axis='index')
    return ddf



if __name__ == '__main__':
    range = query_dataframe('dataset_atsa', first='2015-01-01', last='2017-12-31', tags={'symbol': 'BTC'})
    print(range.head())
    print(range.tail())
