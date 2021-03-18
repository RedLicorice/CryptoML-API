import numpy as np
import pandas as pd


class SplitException(Exception):
    message = ""

    def __init__(self, message):
        self.message = message
        super(SplitException, self).__init__()


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start

            if y is not None:
                beg = indices[start: mid]
                end = indices[mid + margin: stop]
                _y = y
                if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                    _y = y.values
                unique = np.unique(_y[beg:end])
                n_classes = len(unique)
                if n_classes < 2:
                    raise SplitException("Number of classes in fold {} ({}:{}) is invalid! ({}, {})"
                                         .format(i, beg, end, n_classes, unique))
            yield indices[start: mid], indices[mid + margin: stop]
