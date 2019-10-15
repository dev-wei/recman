from sklearn.metrics import roc_auc_score


class RocAucScore:
    def __call__(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def __str__(self):
        return "roc_auc"

    def __repr__(self):
        return "roc_auc"

    @property
    def higher_the_better(self):
        return True
