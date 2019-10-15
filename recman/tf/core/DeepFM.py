import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import log_loss, roc_auc_score

from .DeepModel import DeepModel
from ..inputs import FeatureDictionary
from .layers import (
    DNN,
    DNNCombiner,
    FMLayer,
    FeatEmbeddingLayer,
    PredictionLayer,
    LinearCombiner,
    LinearLayer,
)
# from .utils import (
#     create_feat_inputs,
#     create_loss,
#     create_optimizer,
# )


class DeepFM(DeepModel):
    """
    DeepFM
    https://arxiv.org/abs/1703.04247
    """

    def __init__(
        self,
        feat_dict: FeatureDictionary,
        embedding_size=8,
        embedding_l2_reg=0.00001,
        linear_l2_reg=0.00001,
        fm_dropout=(1.0, 1.0),
        deep_hidden_units=(32, 32),
        deep_dropout=(0.8, 0.8, 0.8),  # good for range (0.6-0.9)
        deep_l2_reg=0.00001,
        deep_activation=tf.nn.relu,
        epoch=10,
        batch_size=64,
        learning_rate=0.001,
        optimizer="adam",
        random_seed=2019,
        use_fm=True,
        use_deep=True,
        loss_type="logloss",
        eval_metric=(roc_auc_score, log_loss),
        what_means_greater=None,
        use_interactive_session=False,
        log_dir="./logs",
    ):
        assert use_fm or use_deep
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
        self.optimizer = optimizer
        self.loss_type = loss_type

        self.embedding_size = embedding_size
        self.embedding_l2_reg = embedding_l2_reg
        self.linear_l2_reg = linear_l2_reg
        self.fm_dropout = fm_dropout
        self.deep_hidden_units = deep_hidden_units
        self.deep_dropout = deep_dropout
        self.deep_l2_reg = deep_l2_reg
        self.deep_activation = deep_activation

        self.use_fm = use_fm
        self.use_deep = use_deep
        self.log_dir = log_dir
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope("inputs"):
                self.inputs.update(create_feat_inputs(self.feat_dict))

                self.hp_fm_dropout = tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="hp_fm_dropout"
                )
                self.hp_deep_dropout = tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="hp_deep_dropout"
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
                self.weights.update(feat_embeds.weights)
                self.feat_embeds, self.feat_bias = (
                    tf.concat(list(self.feat_embeds_dict.values()), axis=1),
                    tf.concat(list(self.feat_bias_dict.values()), axis=1),
                )

            with tf.name_scope("Linear"):
                linear_combiner = LinearCombiner(self.feat_dict)
                linear_inputs = linear_combiner(self.inputs)

                linear = LinearLayer(self.feat_dict, self.linear_l2_reg)
                self.linear_logit = linear(linear_inputs, self.train_phase)
                self.weights.update(linear.weights)

            if self.use_fm:
                with tf.name_scope("FM"):
                    fm = FMLayer(self.hp_fm_dropout)
                    self.fm_logit = fm(self.feat_embeds, self.feat_bias)
                    self.weights.update(fm.weights)

            if self.use_deep:
                with tf.name_scope("DeepNeuralNetwork"):
                    dnn_combiner = DNNCombiner()
                    dnn_input = dnn_combiner(
                        [self.feat_embeds] + self.inputs.dense_inputs
                    )

                    dnn = DNN(
                        self.deep_hidden_units,
                        self.deep_dropout,
                        self.deep_activation,
                        self.deep_l2_reg,
                    )
                    self.dnn_logit = dnn(dnn_input)
                    self.weights.update(dnn.weights)

            with tf.name_scope("DeepFM"):
                if self.use_fm and self.use_deep:
                    self.final_logit = tf.add_n(
                        [self.linear_logit, self.fm_logit, self.dnn_logit]
                    )
                elif self.use_fm:
                    self.final_logit = tf.add(self.linear_logit, self.fm_logit)
                elif self.use_deep:
                    self.final_logit = tf.add(self.linear_logit, self.dnn_logit)

            with tf.name_scope("output"):
                output = PredictionLayer(use_bias=False)
                self.out = output(self.final_logit)
                self.weights.update(output.weights)

            with tf.name_scope("loss"):
                self.loss = create_loss(self.label, self.out, self.loss_type)
                tf.compat.v1.summary.scalar("loss", self.loss)

            with tf.name_scope("l2_reg"):
                l2_loss = feat_embeds.l2()
                tf.compat.v1.summary.scalar("feat_embeds_l2", l2_loss)
                self.loss += l2_loss

                l2_loss = linear.l2()
                tf.compat.v1.summary.scalar("linear_l2", l2_loss)
                self.loss += l2_loss

                if self.use_deep:
                    l2_loss = dnn.l2()
                    tf.compat.v1.summary.scalar("dnn_l2", l2_loss)
                    self.loss += l2_loss

                tf.compat.v1.summary.scalar("loss_after_l2", self.loss)

            with tf.name_scope("Optimizer"):
                self.optimizer = create_optimizer(
                    self.optimizer, self.learning_rate, self.loss
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
            feed_dict[self.hp_fm_dropout] = self.fm_dropout
            feed_dict[self.hp_deep_dropout] = self.deep_dropout
        else:
            feed_dict[self.hp_fm_dropout] = [1] * len(self.fm_dropout)
            feed_dict[self.hp_deep_dropout] = [1] * len(self.deep_dropout)

        feed_dict[self.label] = y
        feed_dict[self.train_phase] = training
        return feed_dict

    def fit_on_batch(self, X, y):
        loss, opt = self.session.run(
            [self.loss, self.optimizer], feed_dict=self.create_inputs(X, y)
        )
