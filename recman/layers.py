import itertools
from collections import OrderedDict
from functools import reduce
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from .utils import glorot_normal, to_tensor, to_sparse_tensor
from .inputs import (
    DenseFeat,
    FeatureDictionary,
    FeatureInputs,
    MultiValCsvFeat,
    MultiValSparseFeat,
    SequenceFeat,
    SparseFeat,
    SparseValueFeat,
)


class LinearCombiner:
    """
    Linear Combiner
    """

    def __init__(self, feat_dict: FeatureDictionary, prefix=""):
        self.feat_dict = feat_dict
        self.prefix = f"{prefix}linear_"
        self.dense_feats = self.feat_dict.dense_feats

    def __call__(self, inputs: FeatureInputs) -> tf.Tensor:
        feat_tensors = []
        offset = 0

        for feat in (
            self.feat_dict.sparse_feats
            + self.feat_dict.sparse_val_feats
            + self.feat_dict.multi_val_csv_feats
        ):
            feat_tensor = tf.cast(to_tensor(feat, inputs[feat]), tf.float32)
            tf.compat.v1.logging.info(
                f"{self.prefix}{feat.name}_offset: {offset},{offset+int(feat_tensor.shape[1])}"
            )
            feat_tensors.append(feat_tensor)
            offset += feat.feat_size

        self.output = tf.concat(feat_tensors + inputs.dense_inputs, axis=1)
        return self.output

    @property
    def output_shape(self):
        return -1, int(self.output.shape[1])


class DNNCombiner:
    """
    DNN Combiner
    """

    def __init__(self):
        pass

    def __call__(self, inputs: list) -> tf.Tensor:
        self.result = tf.concat(
            [tf.compat.v1.layers.Flatten()(each_input) for each_input in inputs], axis=1
        )
        return self.result

    @property
    def output_shape(self):
        return -1, int(self.result.shape[-1])


class ASPCombiner:
    def __init__(self, feat_dict: FeatureDictionary, prefix=""):
        self.prefix = prefix
        self.feat_dict = feat_dict

    def __call__(self, embeds_dict):
        query_embeds = []
        key_embeds = []

        self.field_size = 0
        self.key_size = 0
        self.embedding_size = -1
        for feat in self.feat_dict.sequence_feats:
            id_embed = embeds_dict[f"{self.prefix}{feat.id_feat.name}_feat_embed"]
            key_embed = embeds_dict[f"{self.prefix}{feat.name}_feat_embed"]
            query_embeds.append(id_embed)
            key_embeds.append(key_embed)

            self.field_size += 1
            self.key_size += feat.max_len
            self.embedding_size = int(id_embed.shape[-1])

        return tf.concat(query_embeds, axis=-1), tf.concat(key_embeds, axis=-1)

    @property
    def output_shape(self):
        return (
            (-1, self.field_size, self.embedding_size),
            (-1, self.key_size, self.embedding_size),
        )


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

    def __init__(self, feat, embedding_size, l2_reg=0.00001, prefix=""):
        assert not isinstance(feat, DenseFeat)
        self.feat = feat
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.prefix = prefix

    def _create_weights(self):
        weights = dict()

        if (
            isinstance(self.feat, SparseFeat)
            or isinstance(self.feat, MultiValSparseFeat)
            or isinstance(self.feat, SparseValueFeat)
            or isinstance(self.feat, MultiValCsvFeat)
        ):
            feat_size = self.feat.feat_size
        else:
            return dict()

        name = f"{self.prefix}{self.feat.name}_feat_embed"
        W = tf.Variable(
            glorot_normal((feat_size, self.embedding_size)), dtype=tf.float32, name=name
        )
        weights[name] = W
        tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
        tf.compat.v1.summary.histogram(name, W)

        name = f"{self.prefix}{self.feat.name}_feat_bias"
        B = tf.Variable(tf.zeros((feat_size, 1)), name=name, dtype=tf.float32)
        weights[name] = B
        tf.compat.v1.logging.info(f"{name}: %s" % B.shape)
        tf.compat.v1.summary.histogram(name, B)

        return weights

    def __call__(
        self,
        feat_input,
        ref_embeddings=None,
        train_phase: tf.Tensor = tf.constant(True, dtype=tf.bool),
    ):
        self.weights = self._create_weights()

        feat_embeds_dict, feat_bias_dict = dict(), dict()
        if isinstance(self.feat, SparseFeat) or isinstance(self.feat, SparseValueFeat):
            feat_embeds = tf.nn.embedding_lookup(
                self.weights[f"{self.prefix}{self.feat.name}_feat_embed"],
                feat_input[:, 0],
            )
            feat_bias = tf.nn.embedding_lookup(
                self.weights[f"{self.prefix}{self.feat.name}_feat_bias"],
                feat_input[:, 0],
            )

            if isinstance(self.feat, SparseValueFeat):
                feat_embeds = tf.multiply(feat_embeds, feat_input[:, 1])

            feat_embeds_dict[
                f"{self.prefix}{self.feat.name}_feat_embed"
            ] = tf.expand_dims(feat_embeds, axis=1)
            feat_bias_dict[f"{self.prefix}{self.feat.name}_feat_bias"] = tf.expand_dims(
                feat_bias, axis=1
            )

        elif isinstance(self.feat, MultiValCsvFeat) or isinstance(
            self.feat, MultiValSparseFeat
        ):
            sparse_tensor = to_sparse_tensor(self.feat, feat_input)

            feat_embeds_dict[f"{self.prefix}{self.feat.name}_feat_embed"] = tf.reshape(
                tf.nn.embedding_lookup_sparse(
                    self.weights[f"{self.prefix}{self.feat.name}_feat_embed"],
                    sp_ids=sparse_tensor,
                    sp_weights=None,
                    combiner="sqrtn",
                ),
                shape=[-1, 1, self.embedding_size],
            )
            feat_bias_dict[f"{self.prefix}{self.feat.name}_feat_embed"] = tf.reshape(
                tf.nn.embedding_lookup_sparse(
                    self.weights[f"{self.prefix}{self.feat.name}_feat_bias"],
                    sp_ids=sparse_tensor,
                    sp_weights=None,
                    combiner="sqrtn",
                ),
                shape=[-1, 1, 1],
            )
        elif isinstance(self.feat, SequenceFeat):
            feat_embeds_dict[
                f"{self.prefix}{self.feat.name}_feat_embed"
            ] = tf.nn.embedding_lookup(
                ref_embeddings[f"{self.prefix}{self.feat.id_feat.name}_feat_embed"],
                feat_input,
            )
            feat_bias_dict[
                f"{self.prefix}{self.feat.name}_feat_bias"
            ] = tf.nn.embedding_lookup(
                ref_embeddings[f"{self.prefix}{self.feat.id_feat.name}_feat_bias"],
                feat_input,
            )

        return feat_embeds_dict, feat_bias_dict

    def l2(self):
        if self.l2_reg > 0:
            return tf.multiply(
                self.l2_reg,
                tf.nn.l2_loss(
                    self.weights[f"{self.prefix}{self.feat.name}_feat_embed"]
                ),
            )
        else:
            return tf.constant(0.0)

    @property
    def output_shape(self):
        return (
            -1,
            1 if not isinstance(self.feat, SequenceFeat) else self.feat.max_len,
            self.embedding_size,
        )


class FeatEmbeddingLayer:
    """
    Feature Embeddings, a set contains all input feature embeddings
    """

    def __init__(
        self, feat_dict: FeatureDictionary, embedding_size, l2_reg=0.00001, prefix=""
    ):
        self.feat_dict = feat_dict
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.prefix = prefix

        # TODO: sort the list by putting the Sequence features to the last
        self.feat_embeds = OrderedDict(
            (feat, FeatEmbedding(feat, self.embedding_size, self.l2_reg, prefix=prefix))
            for feat in self.feat_dict.embedding_feats
        )

    def __call__(
        self,
        inputs: FeatureInputs,
        train_phase: tf.Tensor = tf.constant(True, dtype=tf.bool),
    ):
        feat_embeds_dict, feat_bias_dict = dict(), dict()
        for feat in self.feat_embeds:
            if isinstance(feat, SequenceFeat):
                feat_embed, feat_bias = self.feat_embeds[feat](
                    inputs[feat],
                    self.feat_embeds[feat.id_feat].weights,
                    training=train_phase,
                )
            else:
                feat_embed, feat_bias = self.feat_embeds[feat](
                    inputs[feat], train_phase=train_phase
                )
            feat_embeds_dict.update(feat_embed)
            feat_bias_dict.update(feat_bias)

        self.weights, self.field_size = reduce(
            lambda acc, fe: ({**acc[0], **fe.weights}, acc[1] + fe.output_shape[1]),
            self.feat_embeds.values(),
            (dict(), 0),
        )

        return feat_embeds_dict, feat_bias_dict

    def l2(self):
        l2_val = tf.constant(0.0)
        for feat_embed in self.feat_embeds.values():
            l2_val += feat_embed.l2()
        tf.compat.v1.summary.scalar("feat_embeds_l2", l2_val)
        return l2_val

    @property
    def output_shape(self):
        return -1, self.field_size, self.embedding_size


class LinearLayer:
    """
    Linear Layer
    """

    def __init__(self, feat_dict: FeatureDictionary, l2_reg=0.00001, prefix=""):
        self.feat_dict = feat_dict
        self.l2_reg = l2_reg
        self.prefix = prefix

    def _create_weights(self):
        weights = dict()

        name = f"{self.prefix}linear_w0"
        W0 = tf.Variable(tf.zeros([1]), name=name)
        weights[name] = W0
        tf.compat.v1.logging.info(f"{name}: %s" % W0.shape)
        tf.compat.v1.summary.histogram(name, W0)

        name = f"{self.prefix}linear_w"
        W = tf.Variable(tf.zeros((self.input_shape[1],)), name=name)
        weights[name] = W
        tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
        tf.compat.v1.summary.histogram(name, W)

        return weights

    def __call__(self, inputs: tf.Tensor, train_phase) -> tf.Tensor:
        assert len(inputs.shape) == 2

        self.input_shape = (-1, int(inputs.shape[1]))
        self.weights = self._create_weights()

        feat_weights = []
        for feat in (
            self.feat_dict.sparse_feats
            + self.feat_dict.sparse_val_feats
            + self.feat_dict.multi_val_csv_feats
            + self.feat_dict.dense_feats
        ):
            if feat.weights is not None:
                feat_weights.append(tf.constant(feat.weights, dtype=tf.float32))
            else:
                feat_weights.append(tf.zeros(shape=(feat.feat_size,)))
        feat_weights = tf.concat(feat_weights, axis=-1)

        W = self.weights[f"{self.prefix}linear_w"]
        W = tf.cond(train_phase, lambda: W, lambda: tf.add(W, feat_weights))

        return tf.add(
            self.weights[f"{self.prefix}linear_w0"],
            tf.reduce_sum(tf.multiply(W, inputs), axis=1, keepdims=True),
        )

    def l2(self):
        if self.l2_reg > 0:
            l2_val = tf.multiply(
                self.l2_reg, tf.nn.l2_loss(self.weights[f"{self.prefix}linear_w"])
            )
            tf.compat.v1.summary.scalar("linear_l2", l2_val)
            return l2_val
        else:
            return tf.constant(0.0)

    @property
    def output_shape(self):
        return -1, 1


class FMLayer:
    """
    Factorization Machine Layer
    """

    def __init__(self, dropout=(1, 1)):
        self.dropout = dropout
        self.weights = dict()

    def __call__(self, embeddings: tf.Tensor, embedding_bias: tf.Tensor) -> tf.Tensor:
        assert len(embeddings.shape) == 3 and len(embeddings.shape) == 3

        self.input_shape = -1, int(embeddings.shape[1]), int(embeddings.shape[-1])

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

    @property
    def output_shape(self):
        return -1, self.input_shape[1] + self.input_shape[-1]


class DNN:
    """
    Deep Neron Network
    """

    def __init__(self, hidden_units, dropout, activation, l2_reg=0.00001, prefix=""):
        assert len(hidden_units) + 1 == len(dropout)

        self.hidden_units = hidden_units
        self.dropout = dropout
        self.activation = activation
        self.prefix = prefix
        self.l2_reg = l2_reg

    def _create_weights(self):
        weights = dict()

        tf.compat.v1.logging.info(f"{self.prefix}dnn_layer: {len(self.hidden_units)}")

        name = f"{self.prefix}dnn_layer_0_weights"
        W = tf.Variable(
            glorot_normal((self.input_shape[1], self.hidden_units[0])),
            name=name,
            dtype=tf.float32,
        )
        weights[name] = W
        tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
        tf.compat.v1.summary.histogram(name, W)

        name = f"{self.prefix}dnn_layer_0_bias"
        B = tf.Variable(tf.zeros((self.hidden_units[0],)), name=name, dtype=tf.float32)
        weights[name] = B
        tf.compat.v1.logging.info(f"{name}: %s" % B.shape)
        tf.compat.v1.summary.histogram(name, W)

        for i in range(1, len(self.hidden_units)):
            name = f"{self.prefix}dnn_layer_{i}_weights"
            W = tf.Variable(
                glorot_normal((self.hidden_units[i - 1], self.hidden_units[i])),
                name=name,
                dtype=tf.float32,
            )  # layers[i-1] * layers[i]
            weights[name] = W
            tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
            tf.compat.v1.summary.histogram(name, W)

            name = f"{self.prefix}dnn_layer_{i}_bias"
            B = tf.Variable(
                tf.zeros((self.hidden_units[i],)),
                name=f"{self.prefix}dnn_layer_{i}_bias",
                dtype=tf.float32,
            )  # 1 * layer[i]
            weights[name] = B
            tf.compat.v1.logging.info(f"{name}: %s" % B.shape)
            tf.compat.v1.summary.histogram(name, B)

        return weights

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        assert len(inputs.shape) == 2

        self.input_shape = -1, int(inputs.shape[-1])
        self.weights = self._create_weights()

        y_deep = inputs
        y_deep = tf.nn.dropout(y_deep, rate=1 - self.dropout[0])

        for i in range(len(self.hidden_units)):
            y_deep = tf.nn.bias_add(
                tf.tensordot(
                    y_deep,
                    self.weights[f"{self.prefix}dnn_layer_{i}_weights"],
                    axes=(-1, 0),
                ),
                self.weights[f"{self.prefix}dnn_layer_{i}_bias"],
            )

            y_deep = self.activation(y_deep)
            y_deep = tf.nn.dropout(y_deep, rate=1 - self.dropout[i + 1])

        return Dense(units=1)(y_deep)

    def l2(self):
        if self.l2_reg > 0:
            l2_val = tf.constant(0.0)
            for i in range(len(self.hidden_units)):
                l2_val += tf.multiply(
                    self.l2_reg,
                    tf.nn.l2_loss(self.weights[f"{self.prefix}dnn_layer_{i}_weights"]),
                )
            tf.compat.v1.summary.scalar("dnn_l2", l2_val)
            return l2_val
        else:
            return tf.constant(0.0)

    @property
    def output_shape(self):
        return -1, 1


class AFMLayer:
    """
    Attentional Factorization Machines
    """

    def __init__(self, factor, dropout, l2_reg=0.00001, prefix="afm_"):
        self.factor = factor
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.prefix = prefix

    def _create_weights(self):
        weights = dict()

        name = f"{self.prefix}attention_W"
        W = tf.Variable(
            glorot_normal((self.embedding_size, self.factor)),
            dtype=tf.float32,
            name=name,
        )
        weights[name] = W
        tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
        tf.compat.v1.summary.histogram(name, W)

        weights[f"{self.prefix}attention_b"] = tf.Variable(
            tf.zeros((self.factor,)), dtype=tf.float32, name=f"{self.prefix}attention_b"
        )

        weights[f"{self.prefix}projection_h"] = tf.Variable(
            glorot_normal((self.factor, 1)),
            dtype=tf.float32,
            name=f"{self.prefix}projection_h",
        )

        weights[f"{self.prefix}projection_p"] = tf.Variable(
            tf.zeros((self.embedding_size, 1)),
            dtype=tf.float32,
            name=f"{self.prefix}projection_p",
        )

        return weights

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        assert len(inputs.shape) == 3

        self.embedding_size = int(inputs.shape[-1])
        self.weights = self._create_weights()

        field_size = int(inputs.shape[1])
        feat_embeds_list = tf.split(inputs, num_or_size_splits=[1] * field_size, axis=1)

        assert field_size > 1
        row = []
        col = []

        for r, c in itertools.combinations(feat_embeds_list, 2):
            row.append(r)
            col.append(c)

        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        bi_interaction = tf.multiply(p, q)

        att = tf.nn.relu(
            tf.nn.bias_add(
                tf.tensordot(
                    bi_interaction,
                    self.weights[f"{self.prefix}attention_W"],
                    axes=(-1, 0),
                ),
                self.weights[f"{self.prefix}attention_b"],
            )
        )
        normalized_att_score = tf.nn.softmax(
            tf.tensordot(
                bi_interaction, self.weights[f"{self.prefix}projection_h"], axes=(-1, 0)
            ),
            axis=1,
        )
        att_output = tf.reduce_sum(
            tf.multiply(normalized_att_score, bi_interaction), axis=1
        )
        att_output = tf.nn.dropout(att_output, rate=1 - self.dropout)

        return tf.tensordot(
            att_output, self.weights[f"{self.prefix}projection_p"], axes=(-1, 0)
        )

    def l2(self):
        if self.l2_reg > 0:
            return tf.multiply(
                self.l2_reg, tf.nn.l2_loss(self.weights[f"{self.prefix}attention_W"])
            )
        else:
            return tf.constant(0.0)

    @property
    def output_shape(self):
        return -1, 1


class CIN:
    """
    Compressed Interaction Network
    """

    def __init__(
        self,
        cross_layer_units,
        activation=None,
        dropout=None,
        l2_reg=0.00001,
        prefix="",
    ):
        self.cross_layer_units = cross_layer_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.prefix = prefix
        self.dropout = (
            dropout if dropout is not None else [1] * len(cross_layer_units) + 1
        )

        assert len(self.cross_layer_units) > 0
        assert len(self.cross_layer_units) + 1 == len(self.dropout)

    def _create_weights(self):
        weights = dict()

        field_nums = [self.field_size]

        for i, size in enumerate(self.cross_layer_units):
            name = f"{self.prefix}cin_filter_{i}"
            W = tf.Variable(
                glorot_normal((1, field_nums[-1] * field_nums[0], size)),
                name=name,
                dtype=np.float32,
            )
            weights[name] = W
            tf.compat.v1.logging.info(f"{name}: %s" % W.shape)
            tf.compat.v1.summary.histogram(name, W)

            name = f"{self.prefix}cin_bias_{i}"
            B = tf.Variable(
                tf.zeros([size]), name=f"{self.prefix}cin_bias_{i}", dtype=np.float32
            )
            weights[name] = B
            tf.compat.v1.logging.info(f"{name}: %s" % B.shape)
            tf.compat.v1.summary.histogram(name, B)

            if i != len(self.cross_layer_units) - 1:
                field_nums.append(size // 2)
            else:
                field_nums.append(size)

        return weights

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        assert len(inputs.shape) == 3

        self.field_size = int(inputs.shape[1])
        self.weights = self._create_weights()

        embedding_size = int(inputs.shape[-1])
        field_nums = [self.field_size]

        inputs = tf.nn.dropout(inputs, rate=1 - self.dropout[0])
        hidden_nn_layers = [inputs]
        final_results = []

        split_tensor_0 = tf.split(hidden_nn_layers[0], embedding_size * [1], 2)
        for i, size in enumerate(self.cross_layer_units):
            split_tensor = tf.split(hidden_nn_layers[-1], embedding_size * [1], 2)

            dot_result_m = tf.matmul(split_tensor_0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m, shape=[embedding_size, -1, field_nums[0] * field_nums[i]]
            )
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result,
                filters=self.weights[f"{self.prefix}cin_filter_{i}"],
                stride=1,
                padding="VALID",
            )
            curr_out = tf.nn.bias_add(
                curr_out, self.weights[f"{self.prefix}cin_bias_{i}"]
            )
            if self.activation:
                curr_out = self.activation(curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            curr_out = tf.nn.dropout(curr_out, rate=1 - self.dropout[i + 1])

            if i != len(self.cross_layer_units) - 1:
                field_nums.append(size // 2)
                next_hidden, direct_connect = tf.split(curr_out, 2 * [size // 2], 1)
            else:
                field_nums.append(size)
                direct_connect = curr_out
                next_hidden = 0

            final_results.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result, axis=-1, keepdims=False)
        return Dense(units=1)(result)

    def l2(self):
        if self.l2_reg > 0:
            l2_val = tf.constant(0.0)
            for i in range(len(self.cross_layer_units)):
                l2_val += tf.multiply(
                    self.l2_reg,
                    tf.nn.l2_loss(self.weights[f"{self.prefix}cin_filter_{i}"]),
                )
            tf.compat.v1.summary.scalar("cin_l2", l2_val)
            return l2_val
        else:
            return tf.constant(0.0)

    @property
    def output_shape(self):
        return -1, 1


class ASPLayer:
    """
    Attentional Sequence Pooling
    """

    def __init__(
        self, hidden_units, activation, dropout, weight_normalization, prefix=""
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.weight_normalization = weight_normalization
        self.prefix = prefix

    def __call__(self, queries: tf.Tensor, keys: tf.Tensor) -> tf.Tensor:
        lau = LocalActivationUnit(self.hidden_units, self.activation, self.dropout)
        lau_output = lau(queries, keys)
        raise NotImplementedError


class LocalActivationUnit:
    def __init__(self, hidden_units, activation, dropout, l2_reg=0.00001, prefix=""):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.prefix = prefix

    def _create_weights(self):
        weights = dict()

        weights[f"{self.prefix}lau_kernel"] = tf.Variable(
            glorot_normal((self.size, 1)),
            name=f"{self.prefix}lau_kernel",
            dtype=np.float32,
        )
        weights[f"{self.prefix}lau_bias"] = tf.Variable(
            tf.zeros([1]), name=f"{self.prefix}lau_bias"
        )

        return weights

    def __call__(self, queries: tf.Tensor, keys: tf.Tensor):
        assert len(queries.shape) == 3 and len(keys.shape) == 3

        # self.seq_feats_count = int(queries.shape[1])
        # self.seq_feats_size = int(keys.shape[1]) * int(keys.shape[-1])
        self.size = (
            4 * queries.shape[-1]
            if len(self.hidden_units) == 0
            else self.hidden_units[-1]
        )
        self.weights = self._create_weights()

        keys_len = int(keys.shape[1])

        queries = tf.reshape(
            tf.tile(queries, [1, keys_len, 1]),
            (-1, queries.shape[1] * keys_len, queries.shape[-1]),
        )
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

        dnn_combiner = DNNCombiner()
        dnn_input = dnn_combiner([att_input])
        dnn = DNN(self.hidden_units, self.dropout, self.activation, self.l2_reg)
        att_out = dnn(dnn_input)

        return tf.nn.bias_add(
            tf.tensordot(
                att_out, self.weights[f"{self.prefix}lau_kernel"], axes=(-1, 0)
            ),
            self.weights[f"{self.prefix}lau_bias"],
        )

    @property
    def output_shape(self):
        return -1, 1


class OutputLayer:
    def __init__(self, loss_type="logloss", use_bias=True, bias=0.0):
        self.loss_type = loss_type
        self.use_bias = use_bias
        self.bias = bias

    def _create_weights(self):
        weights = dict()

        name = "global_bias"
        B = tf.Variable(tf.constant(self.bias), name=name, dtype=np.float32)
        weights[name] = B
        tf.compat.v1.logging.info(f"{name}: %s" % B.shape)
        tf.compat.v1.summary.histogram(name, B)

        return weights

    def __call__(self, inputs: tf.Tensor):
        assert len(inputs.shape) == 2 and inputs.shape[-1] == 1

        self.input_shape = -1, int(inputs.shape[-1])
        self.weights = self._create_weights()

        output = inputs
        if self.use_bias:
            output = tf.add(inputs, self.weights["global_bias"])
        if self.loss_type == "logloss":
            output = tf.nn.sigmoid(inputs)

        return output

    def l2(self):
        return tf.constant(0.0)

    @property
    def output_shape(self):
        return -1, 1
