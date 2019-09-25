import tensorflow as tf
import numpy as np
from .input import FeatureInputs, SparseFeat, SparseValueFeat, MultiValCsvFeat


def to_tensor(feat, x):
    if isinstance(feat, SparseFeat):
        return _sparse_feat_to_tensor(feat, x)
    elif isinstance(feat, SparseValueFeat):
        return _sparse_val_feat_to_tensor(feat, x)
    elif isinstance(feat, MultiValCsvFeat):
        return _multi_val_csv_to_tensor(feat, x)
    else:
        raise NotImplementedError


def _sparse_feat_to_tensor(feat, x):
    return tf.one_hot(tf.reshape(x, shape=(-1,)), depth=feat.feat_size)


def _sparse_val_feat_to_tensor(feat, x):
    return tf.one_hot(tf.reshape(x[:, 0], shape=(-1,)), depth=feat.feat_size) * x[:, 1]


def _multi_val_csv_to_tensor(feat, x):
    post_tags_str, = tf.io.decode_csv(x, [[""]])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(feat.tag_hash_table.keys())),
            values=tf.constant(list(feat.tag_hash_table.values())),
        ),
        default_value=0,
    )

    split_tags = tf.string_split(tf.reshape(post_tags_str, shape=(-1,)), "|")

    dense_tensor = tf.sparse.to_dense(
        tf.sparse.SparseTensor(
            indices=split_tags.indices,
            values=table.lookup(split_tags.values),
            dense_shape=split_tags.dense_shape,
        )
    )
    flatten_tensor = tf.reshape(
        tf.reduce_sum(
            tf.transpose(
                tf.one_hot(dense_tensor, depth=feat.feat_size, off_value=0),
                perm=[0, 2, 1],
            ),
            axis=2,
        ),
        shape=(-1, feat.feat_size),
    )
    flatten_zeros = tf.zeros_like(flatten_tensor)
    return tf.concat([flatten_zeros[:, :1], flatten_tensor[:, 1:]], axis=1)


def to_sparse_tensor(feat, x):
    if isinstance(feat, MultiValCsvFeat):
        return _multi_val_csv_to_sparse_tensor(feat, x)
    else:
        raise NotImplementedError


def _multi_val_csv_to_sparse_tensor(feat, x):
    post_tags_str, = tf.io.decode_csv(x, [[""]])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(feat.tag_hash_table.keys())),
            values=tf.constant(list(feat.tag_hash_table.values())),
        ),
        default_value=0,
    )

    split_tags = tf.string_split(tf.reshape(post_tags_str, shape=(-1,)), "|")
    return tf.SparseTensor(
        indices=split_tags.indices,
        values=table.lookup(split_tags.values),
        dense_shape=(-1, len(feat.tags)),
    )


def unique_of_2d_list(x):
    import itertools

    return np.unique(np.array(list(itertools.chain.from_iterable(x.tolist()))))


def dense_to_sparse(dense_tensor: tf.Tensor) -> tf.SparseTensor:
    zero = tf.constant(0, dtype=tf.int64)
    where = tf.not_equal(dense_tensor, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense_tensor, indices)
    return tf.SparseTensor(indices, values, dense_shape=dense_tensor.shape)


def calc_fan(weight_shape):
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError()
    return fan_in, fan_out


def he_uniform(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    std = np.sqrt(2 / fan_in)
    return tf.random.truncated_normal(mean=0, stddev=std, shape=weight_shape)


def glorot_normal(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return tf.random.truncated_normal(mean=0, stddev=std, shape=weight_shape)


def glorot_uniform(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return tf.random.uniform(shape=weight_shape, minval=-b, maxval=b)


def create_feat_inputs(feat_dict) -> FeatureInputs:
    # inputs
    inputs = FeatureInputs()

    for feat in feat_dict.values():
        inputs[feat] = tf.compat.v1.placeholder(
            dtype=feat.dtype, shape=feat.get_shape(), name=f"{feat.name}_input"
        )
    return inputs


def feed_feat_inputs(model, X, y, training=True):
    feed_dict = dict()
    for feat in model.feat_dict.values():
        feed_dict[model.inputs[feat]] = feat(X[feat.name])

    feed_dict[model.label] = y
    feed_dict[model.train_phase] = training
    return feed_dict


def initialize_variables(interactive_session=False):
    init = tf.compat.v1.initializers.global_variables()
    table_init = tf.compat.v1.initializers.tables_initializer()
    session = (
        tf.compat.v1.InteractiveSession()
        if interactive_session
        else tf.compat.v1.Session()
    )
    session.run([init, table_init])
    session.as_default()
    return session


def tensor_board(graph, log_dir="logs/"):
    return (
        tf.compat.v1.summary.merge_all(),
        tf.compat.v1.summary.FileWriter(log_dir, graph),
    )


def create_loss(label, out, loss_type):
    if loss_type == "logloss":
        loss = tf.compat.v1.losses.log_loss(label, out)
    elif loss_type == "mse":
        loss = tf.compat.v1.losses.mean_squared_error(label, out)
    else:
        raise ValueError()

    return loss


def create_optimizer(optimizer, learning_rate, loss):
    if optimizer == "adam":
        return tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        ).minimize(loss)
    elif optimizer == "adagrad":
        return tf.compat.v1.train.AdagradOptimizer(
            learning_rate=learning_rate, initial_accumulator_value=1e-8
        ).minimize(loss)
    elif optimizer == "gd":
        return tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        ).minimize(loss)
    elif optimizer == "momentum":
        return tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.95
        ).minimize(loss)
    elif isinstance(optimizer, tf.compat.v1.train.Optimizer):
        return optimizer.minimize(loss)
    else:
        raise ValueError()


def count_parameters():
    total_parameters = 0
    for variable in tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
    ):
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    tf.compat.v1.logging.info(f"parameters: {total_parameters}")
