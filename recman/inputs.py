import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureDictionary(OrderedDict):
    @property
    def embedding_feats(self):
        return [feat for feat in self.values() if not isinstance(feat, DenseFeat)]

    @property
    def sparse_feats(self):
        return [feat for feat in self.values() if isinstance(feat, SparseFeat)]

    @property
    def sparse_val_feats(self):
        return [feat for feat in self.values() if isinstance(feat, SparseValueFeat)]

    @property
    def dense_feats(self):
        return [feat for feat in self.values() if isinstance(feat, DenseFeat)]

    @property
    def multi_val_csv_feats(self):
        return [feat for feat in self.values() if isinstance(feat, MultiValCsvFeat)]

    @property
    def multi_val_sparse_feats(self):
        return [feat for feat in self.values() if isinstance(feat, MultiValSparseFeat)]

    @property
    def sequence_feats(self):
        return [feat for feat in self.values() if isinstance(feat, SequenceFeat)]

    def initialize(self, X):
        for feat in self.values():
            feat.initialize(X[feat.name])

    def save(self):
        pass

    def load(self):
        pass


class FeatureInputs(OrderedDict):
    @property
    def embedding_inputs(self):
        return [self[feat] for feat in self if not isinstance(feat, DenseFeat)]

    @property
    def sparse_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, SparseFeat)]

    @property
    def sparse_val_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, SparseValueFeat)]

    @property
    def multi_val_csv_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, MultiValCsvFeat)]

    @property
    def multi_val_sparse_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, MultiValSparseFeat)]

    @property
    def sequence_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, SequenceFeat)]

    @property
    def dense_inputs(self):
        return [self[feat] for feat in self if isinstance(feat, DenseFeat)]


class MultiValLabelEncoder:
    def __init__(self, encoder=None):
        self.need_fit = False if encoder else True
        self._encoder = encoder if encoder else ResilientLabelEncoder()

    def fit(self, X, y=None):
        from .utils import unique_of_2d_list

        if self.need_fit:
            unique_items = unique_of_2d_list(X)
            self._encoder.fit(unique_items)
        return self

    def transform(self, X):
        return X.apply(self._encoder.transform)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, y):
        return y.apply(self._encoder.inverse_transform)


class ResilientLabelEncoder:
    def __init__(self, null_val="-----"):
        self.null_val = null_val
        self._encoder = LabelEncoder()

    def fit(self, X, y=None):
        self._encoder.fit(X)
        self._encoder.classes_ = np.concatenate(
            (np.array([self.null_val]), self._encoder.classes_), axis=0
        )
        return self

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        all_classes = set(self._encoder.classes_)

        copy = X.to_frame()
        col_name = copy.columns[0]

        copy.loc[~copy[col_name].isin(all_classes), col_name] = self.null_val

        return copy.apply(self._encoder.transform).values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, y):
        return y.apply(self._encoder.inverse_transform)


class SparseFeat:
    """
    Single Sparse Feature for Normalized Field
    """

    def __init__(
        self,
        name,
        feat_size,
        weights=None,
        dtype=tf.int64,
        encoder=None,
        description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.encoder = encoder if encoder else ResilientLabelEncoder()
        self.feat_size = feat_size + 1
        self._weights = weights
        self._weights_cache = None

    @property
    def weights(self):
        if self._weights:
            if self._weights_cache is None:
                ids = self._weights.keys()
                if self.encoder:
                    ids = self.encoder.transform(list(self._weights.keys()))
                weights = np.zeros((self.feat_size,))
                weights[0] = 0
                for idx, val in zip(ids.flatten(), self._weights.values()):
                    weights[idx] = val
                self._weights_cache = weights
            return self._weights_cache
        return self._weights

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def initialize(self, X):
        if self.encoder:
            self.encoder.fit(X)

    def __call__(self, x):
        if self.encoder:
            x = self.encoder.transform(x)

        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, x):
        return self.encoder.inverse_transform(x) if self.encoder else x

    def __str__(self):
        return f"SparseFeat({self.name}, {self.feat_size}, {self.dtype})"


class SparseValueFeat:
    """
    Single Sparse Feature with value for Field with high dimensionality
    """

    def __init__(
        self,
        name,
        feat_size,
        weights=None,
        dtype=tf.int64,
        encoder=None,
        description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.encoder = encoder if encoder else ResilientLabelEncoder()
        self.feat_size = feat_size + 1
        self._weights = weights
        self._weights_cache = None

    @property
    def weights(self):
        if self._weights:
            if self._weights_cache is None:
                ids = self._weights.keys()
                if self.encoder:
                    ids = self.encoder.transform(list(self._weights.keys()))
                weights = np.zeros((self.feat_size,))
                for idx, val in zip(ids.flatten(), self._weights.values()):
                    weights[idx] = val
                self._weights_cache = weights
            return self._weights_cache
        return self._weights

    def initialize(self, X):
        X = np.array(X.tolist()) if isinstance(X, pd.Series) else X
        assert isinstance(X, np.ndarray) and X.shape[1] == self.get_shape()[1]
        if self.encoder:
            x_ids = X[:, 0]
            self.encoder.fit(x_ids)

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 2

    def __call__(self, x):
        x = np.array(x.tolist()) if isinstance(x, pd.Series) else x
        assert isinstance(x, np.ndarray) and x.shape[1] == self.get_shape()[1]
        if self.encoder:
            x_ids = x[:, 0]
            x[:, 0] = self.encoder.transform(x_ids).reshape(x_ids.shape)

        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, x):
        return self.encoder.inverse_transform(x) if self.encoder else x

    def __str__(self):
        return f"SparseValueFeat({self.name}, {self.feat_size}, {self.dtype})"


class DenseFeat:
    def __init__(
        self,
        name,
        weights=None,
        dtype=tf.float32,
        scaler=StandardScaler(),
        description=None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.scaler = scaler if scaler else StandardScaler()
        self.feat_size = 1
        self._weights = weights

    @property
    def weights(self):
        return [self._weights if self._weights is not None else 0]

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def initialize(self, X):
        if self.scaler:
            self.scaler.fit(X.values.reshape(-1, 1))

    def __call__(self, x, train_phase: tf.Tensor = tf.constant(True, dtype=tf.bool)):
        x = np.array(x, dtype=self.dtype.as_numpy_dtype)

        if self.scaler:
            x = self.scaler.transform(x.reshape(-1, 1))

        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def __str__(self):
        return f"DenseFeat({self.name}, {self.feat_size}, {self.dtype})"


class MultiValSparseFeat:
    def __init__(
        self,
        name,
        feat_size,
        max_len=10,
        dtype=tf.int64,
        encoder=MultiValLabelEncoder(),
        description=None,
    ):
        self.name = name
        self.max_len = max_len
        self.dtype = dtype
        self.encoder = encoder if encoder else MultiValLabelEncoder()
        self.description = description
        self.feat_size = feat_size + 1

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, self.max_len

    def initialize(self, X):
        if self.encoder:
            self.encoder.fit(X)

    def __call__(self, x, training=True):
        if self.encoder:
            x = self.encoder.transform(x)

        x = pad_sequences(x, maxlen=self.max_len, padding="post")
        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, x):
        return self.encoder.inverse_transform(x) if self.encoder else x

    def to_sparse_tensor(self, x):
        from .utils import dense_to_sparse

        if self.dtype != tf.string:
            x = tf.as_string(x)

        hash_x = tf.compat.v1.strings.to_hash_bucket_fast(
            x, self.feat_size
        )  # weak hash

        return dense_to_sparse(hash_x)

    def __str__(self):
        return f"MultiValSparseFeat({self.name}, {self.feat_size}, {self.dtype})"


class MultiValCsvFeat:
    def __init__(self, name, tags=(), weights=None, dtype=tf.string, description=None):
        self.name = name
        self.dtype = dtype
        self.description = description

        self.tags = tags
        self.tag_hash_table = dict((tag, idx + 1) for idx, tag in enumerate(self.tags))

        self.feat_size = len(self.tags) + 1
        self._weights = weights
        self._weights_cache = None

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, 1

    def initialize(self, X):
        pass

    def __call__(self, x, training=True):
        return np.array(x).reshape(self.get_shape(for_tf=False))

    @property
    def weights(self):
        if self._weights:
            if self._weights_cache is None:
                self._weights_cache = [0] * self.feat_size
                for tag, weight in self._weights.items():
                    if tag in self.tag_hash_table:
                        self._weights_cache[self.tag_hash_table[tag]] = weight

            return self._weights_cache
        return self._weights

    def __str__(self):
        return f"MultiValCsvFeat({self.name}, {len(self.tags)}, {self.dtype})"


class SequenceFeat:
    def __init__(
        self, name, id_feat: SparseFeat, max_len=10, dtype=tf.int64, description=None
    ):
        assert id_feat

        self.name = name
        self.id_feat = id_feat
        self.max_len = max_len
        self.dtype = dtype
        self.description = description
        self.encoder = (
            MultiValLabelEncoder(self.id_feat.encoder) if self.id_feat.encoder else None
        )

    def get_shape(self, for_tf=True):
        return None if for_tf else -1, self.max_len

    def initialize(self, X):
        pass

    def __call__(self, x):
        if self.encoder:
            x = self.encoder.transform(x)

        x = pad_sequences(x, maxlen=self.max_len, padding="post")
        return x.astype(dtype=self.dtype.as_numpy_dtype).reshape(
            self.get_shape(for_tf=False)
        )

    def decode(self, x):
        return self.encoder.inverse_transform(x - 1) if self.encoder else x

    def __str__(self):
        return f"SequenceFeat({self.name}, {self.id_feat.name}, {self.dtype})"
