import numpy as np
import tensorflow as tf

from .utils import (
    glorot_normal,
    glorot_uniform,
    one_hot,
    to_sparse_tensor,
    compute_hidden_units_s2,
    convert_to_sparse,
)
from .inputs import (
    DenseFeat,
    MultiValCsvFeat,
    MultiValSparseFeat,
    SequenceFeat,
    SparseFeat,
    SparseValueFeat,
)

import logging

log = logging.getLogger(__name__)


class BatchNormalization:
    def __init__(self, epsilon=1e-3, prefix=""):
        self.epsilon = epsilon
        self.prefix = prefix

    def _create_weights(self):
        weights = dict()

        name = f"{self.prefix}scale"
        Scale = tf.Variable(tf.ones([self.units]))
        weights[name] = Scale
        tf.compat.v1.logging.info(f"{name}: %s" % Scale.shape)
        tf.compat.v1.summary.histogram(name, Scale)

        name = f"{self.prefix}beta"
        Beta = tf.Variable(tf.zeros([self.units]))
        weights[name] = Beta
        tf.compat.v1.logging.info(f"{name}: %s" % Beta.shape)
        tf.compat.v1.summary.histogram(name, Beta)

        return weights

    def __call__(self, inputs):
        assert len(inputs.shape) == 2
        self.units = inputs.shape[1]
        self.weights = self._create_weights()

        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        return tf.nn.batch_normalization(
            inputs,
            batch_mean,
            batch_var,
            self.weights[f"{self.prefix}beta"],
            self.weights[f"{self.prefix}scale"],
            self.epsilon,
        )

    @property
    def output_shape(self):
        return -1, self.units


class FeatEmbedding:
    """
    Feature Embedding
    """

    display_name = "FeatEmbedding"

    def __init__(self, variables, feat, embedding_size, l2_reg=0.00001, prefix=""):
        assert not isinstance(feat, DenseFeat)
        self.feat = feat
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.variables = variables

    def _upsert_variables(self):
        name = f"{self.prefix}{self.feat.name}_feat_embed"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                glorot_normal([self.feat.feat_size, self.embedding_size]),
                dtype=tf.float32,
                name=name,
            )

        name = f"{self.prefix}{self.feat.name}_feat_bias"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                tf.zeros([self.feat.feat_size, 1]), name=name, dtype=tf.float32
            )

    def __call__(self, feat_input):
        with tf.name_scope(f"{self.prefix}{self.feat.name}_{self.display_name}"):
            self._upsert_variables()

            if isinstance(self.feat, SparseFeat) or isinstance(
                self.feat, SparseValueFeat
            ):
                feat_embeds = tf.nn.embedding_lookup(
                    self.variables[f"{self.prefix}{self.feat.name}_feat_embed"],
                    feat_input[:, :1],
                    name=f"{self.prefix}{self.feat.name}_embed_lookup",
                )
                feat_bias = tf.nn.embedding_lookup(
                    self.variables[f"{self.prefix}{self.feat.name}_feat_bias"],
                    feat_input[:, :1],
                    name=f"{self.prefix}{self.feat.name}_bias_lookup",
                )

                if isinstance(self.feat, SparseValueFeat):
                    feat_embeds = tf.multiply(feat_embeds, feat_input[:, 1])

            elif isinstance(self.feat, MultiValCsvFeat) or isinstance(
                self.feat, MultiValSparseFeat
            ):
                sparse_tensor = to_sparse_tensor(self.feat, feat_input)

                feat_embeds = tf.reshape(
                    tf.nn.embedding_lookup_sparse(
                        self.variables[f"{self.prefix}{self.feat.name}_feat_embed"],
                        sp_ids=sparse_tensor,
                        sp_weights=None,
                        combiner="sqrtn",
                        name=f"{self.prefix}{self.feat.name}_embed_lookup",
                    ),
                    shape=[-1, 1, self.embedding_size],
                )
                feat_bias = tf.reshape(
                    tf.nn.embedding_lookup_sparse(
                        self.variables[f"{self.prefix}{self.feat.name}_feat_bias"],
                        sp_ids=sparse_tensor,
                        sp_weights=None,
                        combiner="sqrtn",
                        name=f"{self.prefix}{self.feat.name}_bias_lookup",
                    ),
                    shape=[-1, 1, 1],
                )

            elif isinstance(self.feat, SequenceFeat):
                feat_embeds = tf.nn.embedding_lookup(
                    self.variables[f"{self.prefix}{self.feat.id_feat.name}_feat_embed"],
                    feat_input,
                    name=f"{self.prefix}{self.feat.name}_embed_lookup",
                )
                feat_bias = tf.nn.embedding_lookup(
                    self.variables[f"{self.prefix}{self.feat.id_feat.name}_feat_bias"],
                    feat_input,
                    name=f"{self.prefix}{self.feat.name}_bias_lookup",
                )

        return feat_embeds, feat_bias

    def l2(self):
        return tf.multiply(
            self.l2_reg,
            tf.nn.l2_loss(self.variables[f"{self.prefix}{self.feat.name}_feat_embed"]),
            name=f"{self.prefix}{self.feat.name}_l2",
        )


class FeatEmbeddingLayer:
    """
    Feature Embeddings, a set contains all input feature embeddings
    """

    display_name = "FeatEmbeddingLayer"

    def __init__(self, variables, feat_dict, embedding_size, l2_reg=0.00001, prefix=""):
        self.variables = variables
        self.feat_dict = feat_dict
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.prefix = prefix

        # TODO: sort the list by putting the Sequence features to the last
        self.feat_embeds = dict(
            (
                feat,
                FeatEmbedding(
                    self.variables,
                    feat,
                    self.embedding_size,
                    self.l2_reg,
                    prefix=prefix,
                ),
            )
            for feat in self.feat_dict.embedding_feats
        )

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            self.feat_embeds_dict, self.feat_bias_dict = dict(), dict()

            for feat in self.feat_embeds:
                feat_embed, feat_bias = self.feat_embeds[feat](inputs[feat.name])
                self.feat_embeds_dict[feat.name] = feat_embed
                self.feat_bias_dict[feat.name] = feat_bias

            return (
                tf.concat(
                    list(self.feat_embeds_dict.values()),
                    axis=1,
                    name=f"{self.prefix}feat_embeds",
                ),
                tf.concat(
                    list(self.feat_bias_dict.values()),
                    axis=1,
                    name=f"{self.prefix}feat_bias",
                ),
            )

    def l2(self):
        return tf.add_n(
            [feat_embed.l2() for feat_embed in self.feat_embeds.values()],
            name=f"{self.prefix}feat_embeds_l2",
        )


class LinearCombiner:
    """
    Linear Combiner
    """

    display_name = "LinearCombiner"

    def __init__(self, feat_dict, prefix=""):
        self.feat_dict = feat_dict
        self.prefix = prefix
        self.dense_feats = self.feat_dict.dense_feats

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            sparse_feat_tensors = []
            offset = 0

            for feat in (
                self.feat_dict.sparse_feats
                + self.feat_dict.sparse_val_feats
                + self.feat_dict.multi_val_csv_feats
            ):
                feat_tensor = convert_to_sparse(
                    tf.cast(one_hot(feat, inputs[feat.name]), tf.float32)
                )
                sparse_feat_tensors.append(feat_tensor)
                offset += feat.feat_size

            dense_feat_tensors = []
            for dense_input in inputs.dense_inputs(self.feat_dict):
                dense_feat_tensors.append(convert_to_sparse(dense_input))

            self.output = tf.sparse.concat(
                sp_inputs=sparse_feat_tensors + dense_feat_tensors,
                axis=1,
                name=f"{self.prefix}linear_input",
            )
            return self.output

class LinearCombiner2:
    """
    Linear Combiner
    """

    display_name = "LinearCombiner"

    def __init__(self, feat_dict, prefix=""):
        self.feat_dict = feat_dict
        self.prefix = prefix
        self.dense_feats = self.feat_dict.dense_feats

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            feat_tensors = []
            offset = 0

            for feat in (
                    self.feat_dict.sparse_feats
                    + self.feat_dict.sparse_val_feats
                    + self.feat_dict.multi_val_csv_feats
            ):
                feat_tensor = tf.cast(one_hot(feat, inputs[feat.name]), tf.float32)
                feat_tensors.append(feat_tensor)
                offset += feat.feat_size

            dense_feat_tensors = []
            for dense_input in inputs.dense_inputs(self.feat_dict):
                dense_feat_tensors.append(dense_input)

            self.output = tf.concat(
                feat_tensors + dense_feat_tensors,
                axis=1,
                name=f"{self.prefix}linear_input",
            )
            return self.output

class LinearLayer:
    """
    Linear Layer
    """

    display_name = "LinearRegression"

    def __init__(self, variables, feat_dict, l2_reg=0.00001, prefix="", training=True):
        self.variables = variables
        self.feat_dict = feat_dict
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.training = training

    def _upsert_variables(self, input_shape):
        name = f"{self.prefix}linear_w0"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros([1]), name=name)

        name = f"{self.prefix}linear_w"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros([input_shape[1], 1]), name=name)

    def __call__(self, inputs):
        feat_total_size = 0
        for feat in (
            self.feat_dict.sparse_feats
            + self.feat_dict.sparse_val_feats
            + self.feat_dict.multi_val_csv_feats
            + self.feat_dict.dense_feats
        ):
            feat_total_size += feat.feat_size

        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            self._upsert_variables([-1, feat_total_size])

            W = self.variables[f"{self.prefix}linear_w"]
            W0 = self.variables[f"{self.prefix}linear_w0"]

            if not self.training:
                feat_weights = []
                for feat in (
                    self.feat_dict.sparse_feats
                    + self.feat_dict.sparse_val_feats
                    + self.feat_dict.multi_val_csv_feats
                    + self.feat_dict.dense_feats
                ):
                    feat_weights.append(
                        convert_to_sparse(
                            tf.convert_to_tensor(feat.weights, dtype=tf.float32)
                        )
                    )
                feat_weights = tf.sparse.expand_dims(
                    tf.sparse.concat(sp_inputs=feat_weights, axis=-1), axis=1
                )
                W = tf.sparse.add(W, feat_weights)

            return tf.nn.bias_add(tf.sparse.sparse_dense_matmul(inputs, W), W0)

    def l2(self):
        return tf.multiply(
            self.l2_reg,
            tf.nn.l2_loss(self.variables[f"{self.prefix}linear_w"]),
            name=f"{self.prefix}linear_l2",
        )

class LinearLayer2:
    """
    Linear Layer
    """

    display_name = "LinearRegression"

    def __init__(self, variables, feat_dict, l2_reg=0.00001, prefix="", training=True):
        self.variables = variables
        self.feat_dict = feat_dict
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.training = training

    def _upsert_variables(self, input_shape):
        name = f"{self.prefix}linear_w0"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros([1]), name=name)

        name = f"{self.prefix}linear_w"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros([input_shape[1], 1]), name=name)

    def __call__(self, inputs):
        feat_total_size = 0
        for feat in (
                self.feat_dict.sparse_feats
                + self.feat_dict.sparse_val_feats
                + self.feat_dict.multi_val_csv_feats
                + self.feat_dict.dense_feats
        ):
            feat_total_size += feat.feat_size

        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            self._upsert_variables([-1, feat_total_size])

            W = self.variables[f"{self.prefix}linear_w"]
            W0 = self.variables[f"{self.prefix}linear_w0"]

            if not self.training:
                feat_weights = []
                for feat in (
                        self.feat_dict.sparse_feats
                        + self.feat_dict.sparse_val_feats
                        + self.feat_dict.multi_val_csv_feats
                        + self.feat_dict.dense_feats
                ):
                    feat_weights.append(

                            tf.convert_to_tensor(feat.weights, dtype=tf.float32)

                    )
                feat_weights = tf.expand_dims(
                    tf.concat(feat_weights, axis=-1), axis=1
                )
                W = tf.add(W, feat_weights)

            return tf.nn.bias_add(tf.matmul(inputs, W), W0)

    def l2(self):
        return tf.multiply(
            self.l2_reg,
            tf.nn.l2_loss(self.variables[f"{self.prefix}linear_w"]),
            name=f"{self.prefix}linear_l2",
        )

class FMLayer:
    """
    Factorization Machine Layer
    """

    def __init__(self, dropout=(1, 1)):
        self.dropout = dropout

    def __call__(self, embeddings, embedding_bias):
        assert len(embeddings.shape) == 3 and len(embeddings.shape) == 3

        # first order term - mostly bias
        embedding_bias = tf.nn.dropout(embedding_bias, rate=1 - self.dropout[0])
        y_first_order = tf.reduce_sum(embedding_bias, axis=1, keepdims=False)

        # second order term
        # sum-square-part
        embeddings = tf.nn.dropout(embeddings, rate=1 - self.dropout[1])
        sum_embeds = tf.reduce_sum(embeddings, axis=1, keepdims=True)  # None * k
        square_of_sum = tf.square(sum_embeds)  # None * K

        # square-sum-part
        square_embeds = tf.square(embeddings)
        sum_of_square = tf.reduce_sum(square_embeds, axis=1, keepdims=True)  # None * K

        # second order
        y_second_order = 0.5 * tf.subtract(square_of_sum, sum_of_square)
        y_second_order = tf.reduce_sum(y_second_order, axis=2, keepdims=False)

        return tf.add(y_first_order, y_second_order)

    def l2(self):
        return tf.constant(0.0)


class DNNCombiner:
    """
    DNN Combiner
    """

    display_name = "DNNCombiner"

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, inputs: list):
        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            self.result = tf.concat(
                [tf.keras.layers.Flatten()(each_input) for each_input in inputs],
                axis=1,
                name=f"{self.prefix}dnn_input",
            )
            return self.result


class DNN:
    """
    Deep Neron Network
    """

    display_name = "DeepNeuralNetwork"

    def __init__(
        self,
        variables,
        hidden_units,
        dropout,
        activation,
        l2_reg=0.00001,
        prefix="",
        seed=2019,
    ):
        assert len(hidden_units) > 0
        assert len(hidden_units) + 1 == len(dropout)

        self.variables = variables
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.activation = activation
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.seed = seed

    def _create_weights(self, input_shape):
        name = f"{self.prefix}dnn_layer_0_weights"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                glorot_normal([input_shape[1], self.hidden_units[0]], seed=self.seed),
                name=name,
                dtype=tf.float32,
            )

        name = f"{self.prefix}dnn_layer_0_bias"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                tf.zeros([self.hidden_units[0]]), name=name, dtype=tf.float32
            )

        for i in range(1, len(self.hidden_units)):
            name = f"{self.prefix}dnn_layer_{i}_weights"
            if name not in self.variables:
                self.variables[name] = tf.Variable(
                    glorot_normal(
                        [self.hidden_units[i - 1], self.hidden_units[i]], seed=self.seed
                    ),
                    name=name,
                    dtype=tf.float32,
                )

            name = f"{self.prefix}dnn_layer_{i}_bias"
            if name not in self.variables:
                self.variables[name] = tf.Variable(
                    tf.zeros([self.hidden_units[i]]), name=name, dtype=tf.float32
                )  # 1 * layer[i]

        name = f"{self.prefix}dnn_w"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                glorot_normal([self.hidden_units[-1], 1], seed=self.seed),
                name=name,
                dtype=np.float32,
            )

        name = f"{self.prefix}dnn_w0"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros((1,)), name=name)

    def __call__(self, inputs):
        assert len(inputs.shape) == 2

        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            input_shape = -1, int(inputs.shape[-1])
            if any([unit is None for unit in self.hidden_units]):
                auto_hidden_units = compute_hidden_units_s2(
                    len(self.hidden_units), input_shape[-1]
                )
            self.hidden_units = [
                auto_hidden_units[idx] if val is None else val
                for idx, val in enumerate(self.hidden_units)
            ]

            self._create_weights(input_shape)

            y_deep = tf.nn.dropout(
                inputs, rate=1 - self.dropout[0], name="dnn_layer_0_input"
            )
            self.y_deep = inputs
            for i in range(len(self.hidden_units)):
                y_deep = tf.nn.bias_add(
                    tf.matmul(
                        y_deep, self.variables[f"{self.prefix}dnn_layer_{i}_weights"]
                    ),
                    self.variables[f"{self.prefix}dnn_layer_{i}_bias"],
                    name=f"{self.prefix}dnn_layer_{i}_output",
                )

                y_deep = self.activation(y_deep)
                y_deep = tf.nn.dropout(
                    y_deep, rate=1 - self.dropout[i + 1], name=f"dnn_layer_{i+1}_input"
                )

            return tf.nn.bias_add(
                tf.matmul(y_deep, self.variables[f"{self.prefix}dnn_w"]),
                self.variables[f"{self.prefix}dnn_w0"],
            )

    def l2(self):
        return tf.add_n(
            [
                tf.multiply(
                    self.l2_reg,
                    tf.nn.l2_loss(
                        self.variables[f"{self.prefix}dnn_layer_{i}_weights"]
                    ),
                )
                for i in range(len(self.hidden_units))
            ]
            + [
                tf.multiply(
                    self.l2_reg, tf.nn.l2_loss(self.variables[f"{self.prefix}dnn_w"])
                )
            ],
            name=f"{self.prefix}dnn_l2",
        )


class CIN:
    """
    Compressed Interaction Network
    """

    display_name = "CompressedInteractionNetwork"

    def __init__(
        self,
        variables,
        cross_layer_units,
        activation,
        dropout,
        l2_reg=0.00001,
        prefix="",
        seed=2019,
    ):
        self.variables = variables
        self.cross_layer_units = cross_layer_units
        self.activation = activation
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.seed = seed

        assert len(self.cross_layer_units) > 0
        assert len(self.cross_layer_units) + 1 == len(self.dropout)

    def _upsert_variables(self, field_size):
        field_nums = [field_size]
        final_size = 0
        for i, size in enumerate(self.cross_layer_units):
            name = f"{self.prefix}cin_filter_{i}"
            if name not in self.variables:
                self.variables[name] = tf.Variable(
                    glorot_normal(
                        [1, field_nums[-1] * field_nums[0], size], seed=self.seed
                    ),
                    name=name,
                    dtype=np.float32,
                )

            name = f"{self.prefix}cin_bias_{i}"
            if name not in self.variables:
                self.variables[name] = tf.Variable(
                    tf.zeros([size]),
                    name=f"{self.prefix}cin_bias_{i}",
                    dtype=np.float32,
                )

            field_nums.append(size // 2)
            if i != len(self.cross_layer_units) - 1:
                final_size += field_nums[-1]
            else:
                final_size += size

        name = f"{self.prefix}cin_w"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                glorot_uniform([final_size, 1], seed=self.seed),
                name=name,
                dtype=np.float32,
            )

        name = f"{self.prefix}cin_w0"
        if name not in self.variables:
            self.variables[name] = tf.Variable(tf.zeros([1]), name=name)

    def __call__(self, inputs):
        assert len(inputs.shape) == 3

        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            field_size = int(inputs.shape[1])
            embedding_size = int(inputs.shape[-1])
            self._upsert_variables(field_size=field_size)

            field_nums = [field_size]

            inputs = tf.nn.dropout(inputs, rate=1 - self.dropout[0])
            hidden_nn_layers = [inputs]
            final_results = []

            split_tensor_0 = tf.split(
                hidden_nn_layers[0], num_or_size_splits=embedding_size * [1], axis=2
            )
            for i, size in enumerate(self.cross_layer_units):
                split_tensor = tf.split(
                    hidden_nn_layers[-1],
                    num_or_size_splits=embedding_size * [1],
                    axis=2,
                )

                dot_result_m = tf.matmul(split_tensor_0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(
                    dot_result_m,
                    shape=[embedding_size, -1, field_nums[0] * field_nums[i]],
                )
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                feat_map = tf.nn.conv1d(
                    dot_result,
                    filters=self.variables[f"{self.prefix}cin_filter_{i}"],
                    stride=1,
                    padding="VALID",
                )
                feat_map = tf.nn.bias_add(
                    feat_map, self.variables[f"{self.prefix}cin_bias_{i}"]
                )

                feat_map = self.activation(feat_map)
                feat_map = tf.transpose(feat_map, perm=[0, 2, 1])
                feat_map = tf.nn.dropout(feat_map, rate=1 - self.dropout[i + 1])

                field_nums.append(size // 2)
                if i != len(self.cross_layer_units) - 1:
                    next_hidden, direct_connect = tf.split(
                        feat_map, 2 * [field_nums[-1]], 1
                    )
                else:
                    direct_connect = feat_map
                    next_hidden = 0

                final_results.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

            result = tf.concat(final_results, axis=1)
            result = tf.reduce_sum(result, axis=-1, keepdims=False)

            return tf.nn.bias_add(
                tf.matmul(result, self.variables[f"{self.prefix}cin_w"]),
                self.variables[f"{self.prefix}cin_w0"],
            )

    def l2(self):
        return tf.add_n(
            [
                tf.multiply(
                    self.l2_reg,
                    tf.nn.l2_loss(self.variables[f"{self.prefix}cin_filter_{i}"]),
                )
                for i in range(len(self.cross_layer_units))
            ]
            + [
                tf.multiply(
                    self.l2_reg, tf.nn.l2_loss(self.variables[f"{self.prefix}cin_w"])
                )
            ],
            name=f"{self.prefix}cin_l2",
        )


class PredictionLayer:
    display_name = "Prediction"

    def __init__(self, variables, task="classification", use_bias=False, prefix=""):
        self.task = task
        self.use_bias = use_bias
        self.prefix = prefix
        self.variables = variables

    def _create_weights(self):
        name = f"{self.prefix}global_bias"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                tf.zeros([1]), name=name, dtype=np.float32
            )

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.display_name}"):
            output = inputs

            if self.use_bias:
                self._create_weights()
                output = tf.nn.bias_add(
                    output, self.variables[f"{self.prefix}global_bias"]
                )
            if self.task == "classification":
                output = tf.math.sigmoid(output)

            return tf.reshape(output, shape=(-1,))
