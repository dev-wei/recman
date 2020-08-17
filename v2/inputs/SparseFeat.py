import numpy as np
import tensorflow as tf

from .ResilientLabelEncoder import ResilientLabelEncoder


class SparseFeat:
    """
    Single Sparse Feature for Normalized Field
    """

    def __init__(
        self,
        name,
        feat_size,
        dtype=tf.int64,
        normalizer=ResilientLabelEncoder(),
        description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.normalizer = normalizer

        # always adding 1 for the room of null value
        self.feat_size = feat_size + 1

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def set_weights(self, val):
        self._weights = val
        self._weights_cache = None

    def initialize(self, X):
        if self.normalizer:
            self.normalizer.fit(X)

    def __call__(self, X):
        if self.normalizer:
            X = self.normalizer.transform(X)

        return X.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, X):
        return self.normalizer.inverse_transform(X) if self.normalizer else X

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"SparseFeat(name={self.name}, feat_size={self.feat_size}, dtype={self.dtype})"
