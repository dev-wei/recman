import tensorflow as tf
from ..inputs import DenseFeat


class LinearCombiner:
    """
    Linear Combiner
    """

    def __init__(self, linear_feats, prefix=""):
        self.linear_feats = linear_feats
        self.prefix = prefix

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.__class__.__name__}"):
            feat_tensors = []
            for feat in self.linear_feats:
                if isinstance(feat, DenseFeat):
                    feat_tensors.append(inputs[feat.name])
                else:
                    feat_tensor = tf.cast(one_hot(feat, inputs[feat.name]), tf.float32)
                    feat_tensors.append(feat_tensor)

            return tf.concat(feat_tensors, axis=1, name=f"{self.prefix}linear_input")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}(feats={self.linear_feats})"


class LinearLayer:
    """
    Linear Layer
    """

    def __init__(self, variables, linear_feats, l2_reg=0.00001, prefix=""):
        self.variables = variables
        self.linear_feats = linear_feats
        self.l2_reg = l2_reg
        self.prefix = prefix

    def _upsert_variables(self, input_shape):
        name = f"{self.prefix}linear_w0"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                tf.zeros([1]), name=name, dtype=tf.float32
            )

        name = f"{self.prefix}linear_w"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                tf.zeros([input_shape[1], 1]), name=name, dtype=tf.float32
            )

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.__class__.__name__}"):
            feat_total_size = sum([feat.feat_size for feat in self.linear_feats])
            self._upsert_variables([-1, feat_total_size])

            W = self.variables[f"{self.prefix}linear_w"]
            W0 = self.variables[f"{self.prefix}linear_w0"]
            return tf.nn.bias_add(tf.matmul(inputs, W, a_is_sparse=True), W0)

    def l2(self):
        return tf.multiply(
            self.l2_reg,
            tf.nn.l2_loss(self.variables[f"{self.prefix}linear_w"]),
            name=f"{self.prefix}linear_l2",
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}"
