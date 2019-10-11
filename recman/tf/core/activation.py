import tensorflow as tf

def activate(activation, inputs):
    if activation == "dice":
        return Dice()


class Dice:
    def __init__(self, axis=-1, epsilon=1e-9, prefix=""):
        self.axis = -1
        self.epsilon = epsilon
        self.prefix = prefix

    def _create_weights(self):
        return {
            f"{self.prefix}dice_alphas": tf.get_variable(
                f"{self.prefix}dice_alphas",
                inputs.shape[-1],
                initializer=tf.compat.v1.constant_initializer(0.0),
                dtype=tf.float32,
            )
        }

    def __call__(self, inputs: tf.Tensor):

        self.weights = self._create_weights()

        # alphas = self.

        input_shape = list(inputs.shape)
        reduction_axes = list(range(len(input_shape)))

        del reduction_axes[self.axis]

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean = tf.reduce_mean(inputs, axis=reduction_axes)
        broadcast_mean = tf.reshape(mean, broadcast_shape)

        var = tf.reduce_mean(
            tf.square(inputs - broadcast_mean) + self.epsilon, axis=reduction_axes
        )
        std = tf.sqrt(var)
        broadcast_std = tf.reshape(std, broadcast_shape)

        x_normed = tf.layers.batch_normalization(inputs, center=False, scale=False)
        x_p = tf.sigmoid(x_normed)

        return alphas * (1.0 - x_p) * inputs + x_p * inputs
