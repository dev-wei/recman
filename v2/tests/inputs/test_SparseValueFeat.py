from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from v2.inputs import SparseValueFeat

df = pd.DataFrame({"name": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]})


def test_initialize():
    normalizer = MagicMock()
    normalizer.fit = MagicMock()
    normalizer.unique_values_count = len(np.unique(df.name)) + 1
    normalizer.inverse_transform = MagicMock(return_value=df.name.values)
    normalizer.transform = MagicMock(return_value=list(range(len(df))))

    feat = SparseValueFeat(name="name", val_name="val", normalizer=normalizer)
    feat.initialize(df)

    assert normalizer.fit.called
    assert feat.feat_size == normalizer.unique_values_count

    input = feat(df)
    assert np.all(np.isclose(input[:, 0], list(range(len(df)))))
    assert np.all(np.isclose(input[:, 1], df.val.values))

    output = feat.decode(df)
    assert normalizer.inverse_transform.called
    assert np.all(output == df.values)
