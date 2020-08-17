from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from .inputs import FeatDictionary

class DeepModel(BaseEstimator, TransformerMixin, ABC):

    __metaclass__ = ABCMeta

    def __init__(self, feat_dict: FeatDictionary):
        self.feat_dict = feat_dict
        self.variables = dict()

    @abstractmethod
    def fit_on_batch(self, X, y):
        pass
