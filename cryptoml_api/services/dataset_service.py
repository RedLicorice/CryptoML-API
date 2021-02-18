from ..repositories.feature_repository import FeatureRepository
from ..models.classification import SplitClassification, SlidingWindowClassification
from sklearn.model_selection import train_test_split
from cryptoml_common.util.timestamp import add_interval, sub_interval
import pandas as pd

class DatasetService:
    def __init__(self):
        self.repo: FeatureRepository = FeatureRepository()

    def get_dataset(self, symbol, dataset=None, target=None, **kwargs):
        results = []
        if dataset:
            X = self.repo.get_features(dataset, symbol, **kwargs)
            results.append(X)
        if target:
            y = self.repo.get_features(target, symbol, **kwargs)
            results.append(y)
        return pd.concat(results, axis='columns')

    # Typical train-test split, specifying training portion
    # If begin/end kwargs are specified, the result of time filtering is split according
    # to the split parameter.
    def get_classification_split(self, clf: SplitClassification, **kwargs):
        X = self.repo.get_features(clf.dataset, clf.symbol, begin=clf.begin, end=clf.end)
        y = self.repo.get_target(clf.target, clf.symbol,  begin=clf.begin, end=clf.end)
        # If request parameters only includes a subset of the features, select them now
        if clf.features:
            X = X.loc[:, clf.features]
        # Ensure features and targets have the same indices!
        # If |features| > |targets|, "join" over target indices
        if X.shape[0] > y.shape[0]:
            X = X.loc[y.index, :]
        # If |features| < |targets|, "join" over features indices
        elif X.shape[0] < y.shape[0]:
            y = y.loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            shuffle=False,
            train_size=clf.split
        )
        return X_train, X_test, y_train, y_test

    # Sliding window without splits. Does not take into account the number of data points, but
    # the actual time. Default unit for window is 'days', can be changed to any of datetime.timedelta
    # attributes.
    def get_classification_window(self, clf: SlidingWindowClassification, **kwargs):
        # Beginning of the time interval included in the window
        # Following python's standard - [begin, end[ - we add one interval to get 'date' 's data in the
        # test set.
        begin = sub_interval(clf.index, amount=clf.train_window, interval=clf.window_interval)
        end = add_interval(clf.index, amount=clf.test_window, interval=clf.window_interval)
        # Get data from repository
        X = self.repo.get_features(clf.dataset, clf.symbol, begin=begin, end=end, **kwargs)
        y = self.repo.get_target(clf.target, clf.symbol, begin=begin, end=end)
        # If request parameters only includes a subset of the features, select them now
        if clf.features:
            X = X.loc[:, clf.features]
        # Ensure features and targets have the same indices!
        # If |features| > |targets|, "join" over target indices
        if X.shape[0] > y.shape[0]:
            X = X.loc[y.index, :]
        # If |features| < |targets|, "join" over features indices
        elif X.shape[0] < y.shape[0]:
            y = y.loc[X.index]
        # Use train_test_split with integer count. This way we get a hassle-free split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            shuffle=False,
            test_size=clf.test_window
        )
        return X_train, X_test, y_train, y_test
