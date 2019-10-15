from sklearn.metrics import log_loss


class LogLoss:
    def __init__(self, eps=1e-07):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        return log_loss(y_true, y_pred, eps=self.eps)

    def __str__(self):
        return "logloss"

    def __repr__(self):
        return "logloss"

    @property
    def higher_the_better(self):
        return False
