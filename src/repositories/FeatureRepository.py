from ._base import BaseRepository
from ..models import Feature, FeatureGroup, Target
import pandas as pd

class FeatureRepository(BaseRepository):
    def __init__(self):
        BaseRepository.__init__(self)

    # A "Feature" instance represents a single entry of a time series,
    # a pandas series is stored as a set of "Feature" instances
    # indexed by Day, Symbol, Name
    def create_feature(self, day, symbol, name, value):
        item = Feature(
            symbol=symbol,
            name=name,
            day=day,  # Ignore the warning, index must be a DateTimeIndex
            value=value
        )
        self.add(item)
        return item

    # A "Target" instance represents a single entry of a time series,
    #  a pandas series is stored as a set of "Target" instances
    def create_target(self, day, symbol, name, value):
        item = Target(
            symbol=symbol,
            name=name,
            day=day,  # Ignore the warning, index must be a DateTimeIndex
            value=value
        )
        self.add(item)
        return item

    # A "FeatureGroup" is a set of features belonging to the same dataset,
    # ie they represent the columns of a dataframe
    def create_feature_group(self, name, features):
        items = []
        for feature in features:
            item = FeatureGroup(
                name=name,
                feature=feature
            )
            self.add(item)
            items.append(item)
        return items

    # Return names of features in a certain group
    # gets column names for the dataframe
    def get_features_in_group(self, group):
        query = self.session.query(FeatureGroup).filter(
            FeatureGroup.name == group
        )
        return [d.feature for d in query.all()]

    # Get a dataframe with features belonging to the specified group
    def get_features_df_from_group(self, group, symbol, **kwargs):
        columns = self.get_features_in_group(group)
        query = self.session.query(Feature).filter(
            Feature.symbol == symbol,
            Feature.name.in_(columns)
        )
        if kwargs.get('begin') and kwargs.get('end'):
            query = query.filter(
                Feature.day.betweeen(kwargs.get('begin'), kwargs.get('end'))
            )
        df = pd.read_sql(sql=query.statement, con=query.session.bind, parse_dates=['day'])
        df = df.pivot(index='day', columns='name', values='value')
        return df

    # Import a Pandas series (in our case a timeseries) as set of "Feature" instances
    def import_series(self, symbol: str, name: str, series: pd.Series):
        for idx, value in series.iteritems():
            self.create_feature(idx, symbol, name, value)
        self.commit()

    # Impord a Pandas dataframe as set of "Feature" instances
    def import_dataframe(self, group: str, symbol: str, df: pd.DataFrame):
        for c in df.columns:
            self.import_series(symbol, c, df[c])
        self.create_feature_group(group, df.columns)