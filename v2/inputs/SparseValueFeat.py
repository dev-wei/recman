import numpy as np
import pandas as pd
import tensorflow as tf

from .ResilientLabelEncoder import ResilientLabelEncoder


class SparseValueFeat:
    """
    Single Sparse Feature for categorical data with numeric value.
    for example. factors decomposition of a polynomial
    factors: [x1, x2, x3, x4]. converted into
    p1: [f1*x1, 0*x2, 0*x3, 0*x4]
    p2: [0*x1, f2*x2, 0*x3, 0*x4]

    factors, x1, x2, x3, x4
    1,       f1  0   0   0
    2,       0   f2  0   0

    In this case, the input can be stored as 2 columns based table

        factor, value
    p1  x1      f1
    p2  x2      f2
    """

    def __init__(
        self,
        name,
        val_name,
        dtype=tf.int64,
        normalizer=ResilientLabelEncoder(),
        description=None,
    ):
        self.name = name
        self.val_name = val_name
        self.dtype = dtype
        self.description = description
        self.normalizer = normalizer
        self.feat_size = None

        assert self.normalizer

    def initialize(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError

        x_ids = X[self.name].values
        self.normalizer.fit(x_ids)
        self.feat_size = self.normalizer.unique_values_count

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 2

    def __call__(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError

        x_trans = np.zeros(X.shape)
        x_trans[:, 0] = self.normalizer.transform(X[self.name])
        x_trans[:, 1] = X[self.val_name]

        return x_trans.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError

        return np.column_stack(
            [self.normalizer.inverse_transform(X[self.name]), X[self.val_name]]
        ).reshape(self.get_shape(for_tf=False))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"val_name={self.val_name}, "
            f"feat_size={self.feat_size}, "
            f"dtype={self.dtype})"
        )
