from v2.inputs import ResilientLabelEncoder
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize(
    "y", [{"y": ["1", "2"]}, {"y": [1, 2]}],
)
def test_encoding_without_null_values(y):
    df_y = pd.DataFrame(y)

    encoder = ResilientLabelEncoder()
    encoder.fit(df_y.y)
    trans_y = encoder.transform(df_y.y)
    assert all(encoder.inverse_transform(trans_y) == df_y.y)


@pytest.mark.parametrize(
    "y", [{"y": ["1", "2", "3"]}, {"y": [1, 2, 3]}],
)
def test_encoding_with_null_values(y):
    df_y = pd.DataFrame(y)

    encoder = ResilientLabelEncoder(nullable=True)
    encoder.fit(df_y.y[:2])
    trans_y = encoder.transform(df_y.y)
    assert all(trans_y == np.array([1, 2, 0]))
