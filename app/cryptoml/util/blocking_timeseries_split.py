import numpy as np


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
            beg = indices[start: mid]
            end = indices[mid + margin: stop]
            if y is not None:
                unique = np.unique(y[beg:end])
                n_classes = len(unique)
                if n_classes < 2:
                    raise SplitException("Number of classes in fold {} ({}:{}) is invalid! ({}, {})"
                                         .format(i, beg, end, n_classes, unique))
            yield beg, end
