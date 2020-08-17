from collections import OrderedDict

from . import DenseFeat


class FeatDictionary(OrderedDict):
    """
    Packing all Feature Definitions
    """

    @property
    def embedding_feats(self):
        return [feat for feat in self.values() if not isinstance(feat, DenseFeat)]

    def initialize(self, X):
        for feat in self.values():
            feat.initialize(X[feat.name])
