from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd

from v2.inputs import SparseFeat

df = pd.DataFrame({"x": [1, 2, 3, 4]})


def test_initialize():
    normalizer = MagicMock()
    normalizer.fit = MagicMock()
    normalizer.unique_values_count = len(np.unique(df.x)) + 1
    normalizer.inverse_transform = MagicMock(return_value=df.x.values)
    normalizer.transform = MagicMock(return_value=df.x.values)

    feat = SparseFeat("x", normalizer=normalizer)
    feat.initialize(df.x)

    assert normalizer.fit.called
    assert feat.feat_size == normalizer.unique_values_count

    input = feat(df.x)
    assert normalizer.transform.called
    assert np.all(np.isclose(input, df.x.values.reshape((-1, 1))))

    output = feat.decode(df.x)
    assert normalizer.inverse_transform.called
    assert np.all(np.isclose(output, df.x.values))
