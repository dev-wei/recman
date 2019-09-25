import numpy as np


def gini(y_true, y_score):
    assert len(y_true) == len(y_score)

    all_together = np.asarray(
        np.c_[y_true, y_score, np.arange(len(y_score))], dtype=np.float
    )
    all_together = all_together[np.lexsort((all_together[:, 2], -1 * all_together[:1]))]
    total_losses = all_together[:, 0].sum()
    gini_sum = all_together[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(y_true) + 1) / 2
    return gini_sum / len(y_true)


def gini_norm(y_true, y_score):
    return gini(y_true, y_score) / gini(y_true, y_true)
