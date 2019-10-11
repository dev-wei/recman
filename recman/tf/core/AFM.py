import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

from .DeepModel import DeepModel
from .layers import (
    # AFMLayer,
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


class AFM(DeepModel):
    """
    Attentional Factorization Machines

    https://arxiv.org/abs/1708.04617
    """

    def __init__(
        self,
        feat_dict: dict,
        embedding_size=8,
        embedding_l2_reg=0.00001,
        linear_l2_reg=0.00001,
        att_factor=8,
        att_l2_reg=0.00001,
        att_dropout=1,
        epoch=10,
        batch_size=256,
        learning_rate=0.001,
        optimizer="adam",
        random_seed=2019,
        use_deep=True,
        loss_type="logloss",
        eval_metric=(roc_auc_score, log_loss),
        l2_reg=0.1,
        what_means_greater=None,
        use_interactive_session=True,
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
        self.att_factor = att_factor
        self.att_l2_reg = att_l2_reg
        self.att_dropout = att_dropout
        self.linear_l2_reg = linear_l2_reg
        self.l2_reg = l2_reg

        self.log_dir = log_dir
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():

            with tf.name_scope("inputs"):
                # inputs
                self.inputs.update(create_feat_inputs(self.feat_dict))

                self.hp_att_dropout = tf.compat.v1.placeholder(
                    tf.float32, name="hp_att_dropout"
                )

                self.label = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 1], name="label"
                )
                self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")

            with tf.name_scope("embeddings"):
                feat_embeds = FeatEmbeddingLayer(
                    self.feat_dict, self.embedding_size, self.embedding_l2_reg
                )
                self.feat_embeds_dict, self.feat_bias_dict = feat_embeds(
                    self.inputs, self.train_phase
                )
                self.weights.update(feat_embeds.weights)
                self.feat_embeds, self.feat_bias = (
                    tf.concat(list(self.feat_embeds_dict.values()), axis=1),
                    tf.concat(list(self.feat_bias_dict.values()), axis=1),
                )

            with tf.name_scope("linear"):
                linear_combiner = LinearCombiner(self.feat_dict)
                linear_inputs = linear_combiner(self.inputs)

                linear = LinearLayer(self.feat_dict, self.linear_l2_reg)
                self.linear_logit = linear(linear_inputs, self.train_phase)
                self.weights.update(linear.weights)

            with tf.name_scope("afm"):
                afm = AFMLayer(self.att_factor, self.att_dropout)
                self.afm_logit = afm(self.feat_embeds)
                self.weights.update(afm.weights)

            with tf.name_scope("output"):
                output = PredictionLayer(use_bias=False)
                self.out = output(tf.add(self.linear_logit, self.afm_logit))
                self.weights.update(output.weights)

            with tf.name_scope("loss"):
                self.loss = create_loss(self.label, self.out, self.loss_type)

            with tf.name_scope("l2_reg"):
                l2_loss = feat_embeds.l2()
                tf.summary.scalar("feat_embeds_l2", l2_loss)
                self.loss += l2_loss

                l2_loss = linear.l2()
                tf.summary.scalar("linear_l2", l2_loss)
                self.loss += l2_loss

                l2_loss = afm.l2()
                tf.summary.scalar("afm_l2", l2_loss)
                self.loss += l2_loss

                tf.summary.scalar("loss_after_l2", self.loss)

            with tf.name_scope("optimizer"):
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

    def _create_feed_dict(self, X, y, training=True):
        feed_dict = dict()
        for feat in self.feat_dict.values():
            feed_dict[self.inputs[feat]] = feat(X[feat.name])

        if training:
            feed_dict[self.hp_att_dropout] = self.att_dropout
        else:
            feed_dict[self.hp_att_dropout] = 1

        feed_dict[self.label] = y
        feed_dict[self.train_phase] = training
        return feed_dict

    def fit_on_batch(self, X, y):
        loss, opt = self.sess.run(
            [self.loss, self.optimizer], feed_dict=self._create_feed_dict(X, y)
        )
