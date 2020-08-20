import numpy as np
import tensorflow as tf


class DenseFeat:
    def __init__(
        self, name, dtype=tf.float32, normalizer=None, description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.normalizer = normalizer
        self.feat_size = 1

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def initialize(self, X):
        if self.normalizer:
            self.normalizer.fit(X.values.reshape(-1, 1))

    def __call__(self, x):
        x = np.array(x, dtype=self.dtype.as_numpy_dtype)

        if self.normalizer:
            x = self.normalizer.transform(x.reshape(-1, 1))

        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"feat_size={self.feat_size}, "
            f"dtype={self.dtype})"
        )
