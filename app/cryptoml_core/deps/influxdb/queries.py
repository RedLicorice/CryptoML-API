from cryptoml_core.deps.influxdb.client import DictClient, DataFrameClient
import pandas as pd
import logging
from cryptoml_core.exceptions import NotFoundException
__all__ = ("query_meta", "query_first_timestamp", "query_last_timestamp", "query_dataframe", "append_tags")


def parse_type(type):
    if type=='integer':
        return 'int64'
    return type


def append_tags(query, tags):
    result = query
    tagstr = ""
    if tags:
        for tag, value in tags.items():
            if not value:
                continue
            tagstr += "AND {}='{}' ".format(tag, value)
    if tagstr:
        if not "WHERE" in query:
            tagstr = tagstr.replace("AND", "WHERE", 1)
        result += " " + tagstr
    return result

# Returns an empty pd.DataFrame to be used as meta for dask.DataFrame
# delayed building
def query_meta(measure, **kwargs) -> pd.DataFrame:
    query = "SHOW FIELD KEYS FROM {}".format(measure)
    with DictClient() as client:
        res = client.query(query)
        types = {k: parse_type(t) for k, t in res.raw['series'][0]['values']}
        cols = [c for c, t in types.items()]
        df = pd.DataFrame(columns=cols)
        df = df.astype(types)
        if kwargs.get('columns'):
            df = df.loc[:, kwargs.get('columns')]
        return df


def query_first_timestamp(measure, tags=None) -> str:
    query = "SELECT * FROM {}".format(measure)
    query = append_tags(query, tags)
    # if tags:
    #     tagstr = ""
    #     for tag, value in tags.items():
    #         if not value:
    #             continue
    #         tagstr += "AND {}='{}' ".format(tag, value)
    #     if tagstr:
    #         tagstr = tagstr.replace("AND", "WHERE", 1)
    #         query += tagstr
    query += " LIMIT 1"
    with DictClient() as client:
        res = client.query(query)
        timestamp = res.raw['series'][0]['values'][0][0]
        return timestamp


def query_last_timestamp(measure, tags=None) -> str:
    query = "SELECT * FROM {}".format(measure)
    query = append_tags(query, tags)
    # if tags:
    #     query += " WHERE"
    #     for tag, value in tags.items():
    #         if not value:
    #             continue
    #         if not query.endswith("WHERE"):
    #             query += " AND"
    #         query += " {}='{}'".format(tag, value)
    query += " ORDER BY time DESC LIMIT 1"
    with DictClient() as client:
        res = client.query(query)
        timestamp = res.raw['series'][0]['values'][0][0]
        return timestamp


def query_dataframe(measure, first=None, last=None, columns=None, **kwargs)-> pd.DataFrame:
    tags = kwargs.get('tags')
    if not first:
        first = query_first_timestamp(measure=measure, tags=tags)
    if not last:
        last = query_last_timestamp(measure=measure, tags=tags)

    if not columns:
        meta = query_meta(measure=measure, columns=columns)
        columns = ','.join([c for c in meta.columns])
    elif isinstance(columns, list):
        columns = ','.join(columns)

    query = "SELECT {} FROM {}".format(columns, measure)
    query += " WHERE time >= '{}' AND time < '{}'".format(first, last)
    query = append_tags(query, tags)
    query += " ORDER BY time"
    logging.info("InfluxDB Query: {}".format(query))
    # if tags:
    #     for tag, value in tags.items():
    #         if not value:
    #             continue
    #         query += " AND {}='{}'".format(tag, value)
    with DataFrameClient() as client:
        res = client.query(query)
        if not measure in res:
            message = "DataFrame {} not found by {}".format(measure, query)
            print(message)
            raise NotFoundException(message)
        data = res[measure]
        return data


if __name__ == '__main__':
    range = query_dataframe('dataset_atsa', first='2015-01-01', last='2017-12-31', tags={'symbol': 'BTC', 'facet':None})
    print(range.head())
    print(range.tail())
