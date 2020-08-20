import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ResilientLabelEncoder:
    def __init__(self, nullable=False):
        self.nullable = nullable
        self._encoder = LabelEncoder()

    def _get_null_value(self, dtype):
        if np.issubdtype(dtype, np.number):
            return float("-inf")
        else:
            return "-------"

    def fit(self, y):
        self._encoder.fit(y)
        if self.nullable:
            null_value = self._get_null_value(y.dtype)
            self._encoder.classes_ = np.concatenate(
                (np.array([null_value]), self._encoder.classes_), axis=0
            )

    def transform(self, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.nullable:
            y = y.copy()
            y.loc[~y.isin(set(self._encoder.classes_))] = self._get_null_value(y.dtype)
            return self._encoder.transform(y)
        else:
            return self._encoder.transform(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return self._encoder.inverse_transform(y)

    @property
    def unique_values_count(self):
        return len(self._encoder.classes_)
