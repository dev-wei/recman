import numpy as np
import tensorflow as tf
import itertools
import numpy as np

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


def glorot_normal(weight_shape, gain=1.0, seed=2019):
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return tf.random.truncated_normal(mean=0, stddev=std, shape=weight_shape, seed=seed)


def unique_of_2d_list(x):
    return np.unique(np.array(list(itertools.chain.from_iterable(x.tolist()))))
