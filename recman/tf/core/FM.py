import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

from .DeepModel import DeepModel
from .layers import (
    FMLayer,
    FeatEmbeddingLayer,
    LinearCombiner,
    LinearLayer,
    PredictionLayer,
)
from .utils import (
    # create_feat_inputs,
    create_loss,
    create_optimizer,
)


class FM(DeepModel):
    """
    Factorization Machine
    """

    def __init__(
        self,
        feat_dict: dict,
        embedding_size=8,
        embedding_l2_reg=0.00001,
        linear_l2_reg=0.00001,
        fm_dropout=(1.0, 1.0),
        epoch=10,
        batch_size=64,
        learning_rate=0.001,
        optimizer="adam",
        random_seed=2019,
        loss_type="logloss",
        eval_metric=(roc_auc_score, log_loss),
        what_means_greater=None,
        use_interactive_session=False,
        log_dir="./logs",
    ):
        assert loss_type in [
            "logloss",
            "mse",
        ], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        DeepModel.__init__(
            self,
            feat_dict,
            epoch,
            batch_size,
            random_seed,
            eval_metric,
            what_means_greater,
            use_interactive_session,
        )

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.loss_type = loss_type

        self.embedding_size = embedding_size
        self.embedding_l2_reg = embedding_l2_reg
        self.fm_dropout = fm_dropout
        self.linear_l2_reg = linear_l2_reg

        self.log_dir = log_dir
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope("FeatureInputs"):
                self.inputs.update(create_feat_inputs(self.feat_dict))

                self.param_fm_dropout = tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="param_fm_dropout"
                )
                self.label = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 1], name="label"
                )
                self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")

            with tf.name_scope("Embeddings"):
                feat_embeds = FeatEmbeddingLayer(
                    self.feat_dict, self.embedding_size, self.embedding_l2_reg
                )
                self.feat_embeds_dict, self.feat_bias_dict = feat_embeds(
                    self.inputs, self.train_phase
                )
                self.feat_embeds, self.feat_bias = (
                    tf.concat(list(self.feat_embeds_dict.values()), axis=1),
                    tf.concat(list(self.feat_bias_dict.values()), axis=1),
                )
                self.weights.update(feat_embeds.weights)

            with tf.name_scope("Linear"):
                linear_combiner = LinearCombiner(self.feat_dict)
                linear_inputs = linear_combiner(self.inputs)

                linear = LinearLayer(self.feat_dict, self.linear_l2_reg)
                self.linear_logit = linear(linear_inputs, self.train_phase)
                self.weights.update(linear.weights)

            with tf.name_scope("FM"):
                fm = FMLayer(self.param_fm_dropout)
                self.fm_logit = fm(self.feat_embeds, self.feat_bias)
                self.weights.update(fm.weights)

            with tf.name_scope("Output"):
                output = PredictionLayer(use_bias=False)
                self.out = output(tf.add(self.fm_logit, self.linear_logit))
                self.weights.update(output.weights)

            with tf.name_scope("Loss"):
                self.loss = create_loss(self.label, self.out, self.loss_type)
                tf.compat.v1.summary.scalar("loss", self.loss)

            with tf.name_scope("L2"):
                l2_loss = feat_embeds.l2()
                tf.compat.v1.summary.scalar("feat_embeds_l2", l2_loss)
                self.loss += l2_loss

                l2_loss = linear.l2()
                tf.compat.v1.summary.scalar("linear_l2", l2_loss)
                self.loss += l2_loss

                tf.compat.v1.summary.scalar("loss_after_l2", self.loss)

            with tf.name_scope("Optimizer"):
                self.optimizer = create_optimizer(
                    self.optimizer_type, self.learning_rate, self.loss
                )

            self.sess = initialize_variables(self.interactive_session)
            self.tb_ops, self.tb_writer = tensor_board(self.graph, log_dir=self.log_dir)

            count_parameters()

    @property
    def tb(self):
        return self.tb_ops, self.tb_writer

    @property
    def session(self):
        return self.sess

    @property
    def output(self):
        return self.out

    def create_inputs(self, X, y, training=True):
        feed_dict = dict()
        for feat in self.feat_dict.values():
            feed_dict[self.inputs[feat]] = feat(X[feat.name])

        if training:
            feed_dict[self.param_fm_dropout] = self.fm_dropout
        else:
            feed_dict[self.param_fm_dropout] = [1] * len(self.fm_dropout)

        feed_dict[self.label] = y
        feed_dict[self.train_phase] = training
        return feed_dict

    def fit_on_batch(self, X, y):
        loss, opt = self.session.run(
            [self.loss, self.optimizer], feed_dict=self.create_inputs(X, y)
        )
