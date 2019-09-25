import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, roc_auc_score

from .DeepModel import DeepModel
from .layers import FeatEmbeddingLayer, ASPCombiner, ASPLayer
from .utils import create_feat_inputs


class DIN(DeepModel):
    """
    Deep Interest Network
    https://arxiv.org/pdf/1706.06978.pdf
    """

    def __init__(
        self,
        feat_dict: dict,
        embedding_size=8,
        att_hidden_units=(80, 40),
        att_activation="dice",
        att_dropout=(1, 1, 1),
        att_weight_normalization=False,
        deep_hidden_units=(32, 32),
        deep_dropout=(0.6, 0.6, 0.6),  # good for range (0.6-0.9)
        deep_l2_reg=0.0,
        deep_activation=tf.nn.relu,
        epoch=10,
        batch_size=256,
        learning_rate=0.001,
        optimizer="adam",
        random_seed=2019,
        loss_type="logloss",
        eval_metric=((roc_auc_score, "score"), (precision_score, "pred")),
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

        self.embedding_size = embedding_size

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.att_dropout = att_dropout
        self.att_weight_normalization = att_weight_normalization

        self.deep_hidden_units = deep_hidden_units
        self.deep_dropout = deep_dropout
        self.deep_activation = deep_activation
        self.l2_reg_deep = deep_l2_reg

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.loss_type = loss_type
        self.l2_reg = l2_reg

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.compat.v1.set_random_seed(self.random_seed)

        with tf.name_scope("inputs"):
            # inputs
            self.inputs.update(create_feat_inputs(self.feat_dict))
            self.label = tf.compat.v1.placeholder(
                tf.float32, shape=[None, 1], name="label"
            )
            self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")

        with tf.name_scope("embedding"):
            feat_embeds = FeatEmbeddingLayer(self.feat_dict, self.embedding_size)
            self.feat_embeds_dict, self.feat_bias_dict = feat_embeds(self.inputs)
            self.feat_embeds, self.feat_bias = (
                tf.concat(list(self.feat_embeds_dict.values()), axis=1),
                tf.concat(list(self.feat_bias_dict.values()), axis=1),
            )
            self.weights.update(feat_embeds.weights)

        # Attention Sequence Pooling
        with tf.name_scope("asp"):
            asp_combiner = ASPCombiner(self.feat_dict)
            self.asp_queries, self.asp_keys = asp_combiner(self.feat_embeds_dict)
            asp = ASPLayer(
                self.att_hidden_units,
                self.att_activation,
                self.att_dropout,
                weight_normalization=self.att_weight_normalization,
            )
            asp_output = asp(self.asp_queries, self.asp_keys)

    def _create_feed_dict(self, X, y):
        feed_dict = dict()
        for feat_name in self.feat_dict:
            feed_dict[self.inputs[f"{feat_name}_input"]] = X[feat_name].values

        feed_dict[self.label] = y
        feed_dict[self.train_phase] = True
        return feed_dict

    def fit_on_batch(self, X, y):
        loss, opt = self.sess.run(
            [self.loss, self.optimizer], feed_dict=self._create_feed_dict(X, y)
        )

    def predict(self, X):
        dummy_y = [1] * len(X)
        batch_index = 0
        X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)
        y_pred = None

        while len(X_batch):
            num_batch = len(y_batch)

            batch_out = self.sess.run(
                self.out, feed_dict=self.Z(X, y_batch)
            )
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)

        return y_pred
