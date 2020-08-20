import numpy as np
import tensorflow as tf

from .ResilientLabelEncoder import ResilientLabelEncoder


class SparseFeat:
    """
    Single Sparse Feature for categorical data.

    for example. user_id, product_id belong to this field type
    user_id: [a1, b2, c3, d4]. converted into
    user_row, a1, b2, c3, d4
    1,        x
    2,            x
    3,                x
    4,                    x
    """

    def __init__(
        self,
        name,
        dtype=tf.int64,
        normalizer=ResilientLabelEncoder(),
        description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.normalizer = normalizer
        self.feat_size = None

        assert self.normalizer

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def initialize(self, X):
        self.normalizer.fit(X)
        self.feat_size = self.normalizer.unique_values_count

    def __call__(self, X):
        X = self.normalizer.transform(X)
        return X.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, X):
        return self.normalizer.inverse_transform(X)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, feat_size={self.feat_size}, dtype={self.dtype})"
