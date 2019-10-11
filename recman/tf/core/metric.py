import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


class LogLoss:
    def __init__(self, eps=1e-07):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        return log_loss(y_true, y_pred, eps=self.eps)

    def __str__(self):
        return "logloss"

    def __repr__(self):
        return "logloss"


class RocAucScore:
    def __call__(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def __str__(self):
        return "roc_auc"

    def __repr__(self):
        return "roc_auc"


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
