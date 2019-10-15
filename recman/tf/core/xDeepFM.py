import tensorflow as tf

from .DeepModel import DeepModel
from .layers import (
    CIN,
    DNN,
    DNNCombiner,
    FeatEmbeddingLayer,
    SparseLinearCombiner,
    SparseLinearLayer,
    # LinearCombiner,
    # LinearLayer,
    PredictionLayer,
)
from .utils import create_loss, create_optimizer, get_linear_features
from ..inputs import FeatureDictionary, DataInputs
from ..hparams.xDeepFM import xDeepFM as HyperParams


class xDeepFM(DeepModel):
    """
    xDeepFM
    https://arxiv.org/pdf/1803.05170.pdf
    """

    def __init__(
            self,
            feat_dict: FeatureDictionary,
            hparams: dict,
            task="classification",
            metrics=(),
            epoch=10,
            batch_size=64,
            random_seed=2019,
    ):
        DeepModel.__init__(
            self,
            feat_dict=feat_dict,
            hparams=hparams,
            epoch=epoch,
            batch_size=batch_size,
            random_seed=random_seed,
            metrics=metrics,
            task=task,
        )

    @tf.function(autograph=False)
    def _out(self, inputs, training=True):
        self.embeddings = FeatEmbeddingLayer(
            self.variables,
            self.feat_dict,
            self.hparams[HyperParams.EmbeddingSize],
            self.hparams[HyperParams.EmbeddingL2Reg],
            use_bias=False,
            seed=self.random_seed,
        )
        feat_embeds, feat_bias = self.embeddings(inputs)

        linear_feats = get_linear_features(
            self.feat_dict, self.hparams[HyperParams.LinearFeatures]
        )
        linear_combiner = SparseLinearCombiner(linear_feats)
        linear_inputs = linear_combiner(inputs)

        self.linear = SparseLinearLayer(
            self.variables,
            linear_feats,
            self.hparams[HyperParams.LinearL2Reg],
            training=training,
        )
        linear_logit = self.linear(linear_inputs)

        self.cin = CIN(
            self.variables,
            self.hparams[HyperParams.CinCrossLayerUnits],
            self.hparams[HyperParams.CinActivation],
            self.hparams[HyperParams.CinDropOut]
            if training
            else [1] * len(self.hparams[HyperParams.CinDropOut]),
            self.hparams[HyperParams.CinL2Reg],
            seed=self.random_seed,
        )
        cin_logit = self.cin(feat_embeds)

        dnn_combiner = DNNCombiner()
        dnn_input = dnn_combiner([feat_embeds] + inputs.dense_inputs(self.feat_dict))

        self.dnn = DNN(
            self.variables,
            self.hparams[HyperParams.DeepHiddenUnits],
            self.hparams[HyperParams.DeepDropOut]
            if training
            else [1] * len(self.hparams[HyperParams.DeepDropOut]),
            self.hparams[HyperParams.DeepActivation],
            self.hparams[HyperParams.DeepL2Reg],
        )
        dnn_logit = self.dnn(dnn_input)

        with tf.name_scope("xDeepFM"):
            final_logit = tf.add_n(
                [linear_logit + cin_logit + dnn_logit], name="final_logit"
            )

        return PredictionLayer(self.variables, self.task)(final_logit)

    def _loss(self, inputs):
        loss = create_loss(inputs.y, self._out(inputs), task=self.task)
        return tf.math.add_n(
            [loss]
            + [
                layer.l2()
                for layer in [self.embeddings, self.linear, self.dnn, self.cin]
            ]
        )

    def fit_on_batch(self, X, y):
        inputs = DataInputs()
        inputs.load(self.feat_dict, X, y)

        # fmt: off
        optimizer = create_optimizer(
            self.hparams[HyperParams.Optimizer],
            self.hparams[HyperParams.LearningRate]
        )
        # fmt: on
        optimizer.minimize(lambda: self._loss(inputs), self.variables.values())
