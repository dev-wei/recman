import tensorflow as tf
from .BaseHyperParameters import BaseHyperParameters


class xDeepFM(BaseHyperParameters):

    EmbeddingSize = "embedding_size"
    EmbeddingL2Reg = "embedding_l2_reg"
    LinearL2Reg = "linear_l2_reg"
    LinearFeatures = "linear_features"
    DeepHiddenUnits = "deep_hidden_units"
    DeepDropOut = "deep_dropout"
    DeepActivation = "deep_activation"
    DeepL2Reg = "deep_l2_reg"
    CinCrossLayerUnits = "cin_cross_layer_units"
    CinDropOut = "cin_dropout"
    CinActivation = "cin_activation"
    CinL2Reg = "cin_l2_reg"

    def __init__(self):
        BaseHyperParameters.__init__(self)

        self.add_param(self.EmbeddingSize, 8)
        self.add_param(self.EmbeddingL2Reg, 0.00001)
        self.add_param(self.LinearL2Reg, 0.00001)
        self.add_param(self.LinearFeatures, [])
        self.add_param(self.DeepHiddenUnits, (32, 32))
        self.add_param(self.DeepDropOut, (0.8, 0.8, 0.8))
        self.add_param(self.DeepActivation, tf.nn.leaky_relu)
        self.add_param(self.DeepL2Reg, 0.00001)
        self.add_param(self.CinCrossLayerUnits, [100, 100, 100])
        self.add_param(self.CinDropOut, [1, 1, 1, 1])
        self.add_param(self.CinActivation, tf.nn.leaky_relu)
        self.add_param(self.CinL2Reg, 0.00001)
