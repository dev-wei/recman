import logging
import numpy as np
import tensorflow as tf

from ..inputs import MultiValCsvFeat, SparseFeat, SparseValueFeat


log = logging.getLogger(__name__)


def split_train_test(df_all, frac=0.8, random_seed=2019):
    df_X_train = df_all.sample(frac=frac, random_state=random_seed)
    df_X_rest = df_all.drop(df_X_train.index)
    df_X_valid = df_X_rest.sample(frac=0.5, random_state=random_seed)
    df_X_test = df_X_rest.drop(df_X_valid.index)

    df_X_train.info()
    df_X_valid.info()
    df_X_test.info()

    log.info(df_X_train.LABEL.describe())
    log.info(df_X_valid.LABEL.describe())
    log.info(df_X_test.LABEL.describe())
    return df_X_train, df_X_valid, df_X_test


def get_linear_features(feat_dict, linear_feats):
    if linear_feats:
        return [feat_dict[feat_name] for feat_name in linear_feats.split(",")]
    else:
        return (
                feat_dict.sparse_feats
                + feat_dict.sparse_val_feats
                + feat_dict.multi_val_csv_feats
                + feat_dict.dense_feats
        )


def convert_to_sparse_tensor(dense_tensor) -> tf.SparseTensor:
    indices = tf.where(
        condition=tf.not_equal(
            x=dense_tensor, y=tf.constant(0, dtype=dense_tensor.dtype)
        )
    )
    values = tf.gather_nd(dense_tensor, indices)
    return tf.sparse.SparseTensor(
        indices, values, dense_shape=tf.shape(dense_tensor, out_type=tf.int64)
    )


def one_hot(feat, x):
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


def _split_tags(feat, x):
    post_tags_str, = tf.io.decode_csv(x, [[""]])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(feat.tag_hash_table.keys())),
            values=tf.constant(list(feat.tag_hash_table.values())),
        ),
        default_value=0,
    )
    return (
        tf.strings.split(tf.reshape(post_tags_str, shape=(-1,)), "|").to_sparse(),
        table,
    )


def _multi_val_csv_to_tensor(feat, x):
    split_tags, table = _split_tags(feat, x)

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
    split_tags, table = _split_tags(feat, x)
    return tf.SparseTensor(
        indices=split_tags.indices,
        values=table.lookup(split_tags.values),
        dense_shape=(-1, len(feat.tags)),
    )


def compute_hidden_units_s1(num_hidden_layers, input_neurons, output_neurons=1):
    r = (input_neurons + output_neurons) ** (1 / (num_hidden_layers + 1))
    hidden_units = []
    for i in range(num_hidden_layers, 0, -1):
        hidden_units.append(round(output_neurons * (r ** i)))

    return hidden_units


def compute_hidden_units_s2(num_hidden_layers, input_neurons, output_neurons=1):
    return [
        round((input_neurons + output_neurons) * 2 / 3)
        for _ in range(num_hidden_layers)
    ]


def unique_of_2d_list(x):
    import itertools

    return np.unique(np.array(list(itertools.chain.from_iterable(x.tolist()))))


def dense_to_sparse(dense_tensor):
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


def he_normal(weight_shape, seed=2019):
    fan_in, fan_out = calc_fan(weight_shape)
    std = np.sqrt(2 / fan_in)
    return tf.random.truncated_normal(mean=0, stddev=std, shape=weight_shape, seed=seed)


def glorot_normal(weight_shape, gain=1.0, seed=2019):
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return tf.random.truncated_normal(mean=0, stddev=std, shape=weight_shape, seed=seed)


def glorot_uniform(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return tf.random.uniform(shape=weight_shape, minval=-b, maxval=b)


def create_loss(y_true, y_pred, task):
    if task == "classification":
        return tf.losses.binary_crossentropy(y_true, y_pred)
    elif task == "regression":
        return tf.losses.mean_squared_error(y_true, y_pred)
    else:
        raise ValueError()


def create_optimizer(optimizer, learning_rate):
    if optimizer == "adam":
        return tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        return tf.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == "gd":
        return tf.optimizers.GradientDescent(learning_rate=learning_rate)
    elif optimizer == "momentum":
        return tf.optimizers.Momentum(learning_rate=learning_rate)
    elif isinstance(optimizer, tf.optimizers.Optimizer):
        return optimizer
    else:
        raise ValueError()


def count_parameters(variables):
    total_parameters = 0
    for variable in variables.values():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


def wrap_up(run_name):
    import boto3
    from zipfile import ZipFile
    from yilia.configuration import S3_BUCKET

    s3 = boto3.client("s3")

    file_paths = [
        "ckpt_model-1.data-00000-of-00001",
        "ckpt_model-1.index",
        "feat_dict",
        "hparams",
        "df_all",
    ]
    zip_file = "deep.latest"
    with ZipFile(zip_file, "w") as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)

    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(zip_file, S3_BUCKET, f"models/deep.latest")
    s3.meta.client.upload_file(zip_file, S3_BUCKET, f"models/deep.{run_name}")
