import tensorflow as tf
import numpy as np


weight_init = tf.random_normal_initializer()
bias_init = tf.constant_initializer(0)


def conv(input, out_channels, filter_size=3, k=1, padding='SAME', mode=None, output_shape=None):
    if tf.rank(input) == 3: input = tf.expand_dims(input, [3])

    in_shape = tf.shape(input)
    input_channels = int(input.get_shape()[1])

    if mode == 'upscale' or mode == 'transpose':
        filter_shape = [filter_size, filter_size, out_channels, input_channels]
    else:
        filter_shape = [filter_size, filter_size, input_channels, out_channels]

    filter = tf.get_variable('filter', filter_shape, initializer=weight_init)
    fan_in = filter_size ** 2 * input_channels
    filter = filter * tf.sqrt(2 / fan_in)

    b = tf.get_variable('bias', [1, out_channels, 1, 1], initializer=bias_init)

    if mode == 'upscale':
        filter = tf.pad(filter, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        filter = tf.add_n([filter[1:, 1:], filter[:-1, 1:], filter[1:, :-1], filter[:-1, :-1]])
        output_shape = [in_shape[0], out_channels, in_shape[2] * 2, in_shape[3] * 2]
        output = tf.nn.conv2d_transpose(input, filter, output_shape, [1, 1, 2, 2],
            padding=padding, data_format='NCHW')

    elif mode == 'downscale':
        filter = tf.pad(filter, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        filter = tf.add_n([filter[1:, 1:], filter[:-1, 1:], filter[1:, :-1], filter[:-1, :-1]])
        filter *= 0.25
        output = tf.nn.conv2d(input, filter, [1, 1, 2, 2], padding=padding, data_format='NCHW')

    elif mode == 'transpose':
        output = tf.nn.conv2d_transpose(input, filter, output_shape, [1, 1, k, k],
            padding=padding, data_format='NCHW')

    else:
        output = tf.nn.conv2d(input, filter, [1, 1, k, k], padding=padding, data_format='NCHW')

    output += b

    if out_channels == 1:
        output = tf.squeeze(output, 3)

    return output


def dense(input, output_size):
    fan_in = int(input.get_shape()[1])
    W = tf.get_variable('W', [fan_in, output_size], initializer=weight_init)
    W = W * tf.sqrt(2 / fan_in)
    b = tf.get_variable('b', [1, output_size, 1, 1], initializer=bias_init)
    return tf.matmul(input, W) + b


def leaky_relu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)


def pixelwise_norm(input):
    pixel_var = tf.reduce_mean(tf.square(input), 1, keepdims=True)
    return input / tf.sqrt(pixel_var + 1e-8)


def conv_layer(input, out_channels, **kwargs):
    return pixelwise_norm(leaky_relu(conv(input, out_channels, **kwargs)))


def decrese_res(input, k=2, data_format='NCHW'):
    filter = [1, 1, k, k]
    pool =  tf.nn.avg_pool(
        input, ksize=filter, strides=filter, padding='SAME', data_format=data_format
    )
    shape = tf.shape(pool)
    output = tf.reshape(pool, [-1, shape[1], shape[2], 1, shape[3], 1])
    output = tf.tile(output, [1, 1, 1, k, 1, k])
    return tf.reshape(output, [-1, shape[1], shape[2] * k, shape[3] * k])


def minibatch_stddev(input):
    shape = tf.shape(input)
    x_ = tf.tile(tf.reduce_mean(input, 0, keepdims=True), [shape[0], 1, 1, 1])
    sigma = tf.sqrt(tf.reduce_mean(tf.square(input - x_), 0, keepdims=True) + 1e-8)
    sigma_avg = tf.reduce_mean(sigma, keepdims=True)
    layer = tf.tile(sigma_avg, [shape[0], shape[1], shape[2], 1])
    return tf.concat((input, layer), 3)


def resize_images(input, dims=None):
    if dims is None:
        dims = tf.shape(input)[2] * 2, tf.shape(input)[3] * 2
    return tf.image.resize_nearest_neighbor(input, dims)


def scale_uint8(input):
    input = tf.to_float(input)
    return (input / 127.5) - 1


def tensor_to_imgs(input, switch_dims=True):
    if switch_dims: input = tf.transpose(input, (0, 2, 3, 1))
    imgs = tf.minimum(tf.maximum(input, -tf.ones_like(input)), tf.ones_like(input))
    imgs = (imgs + 1) * 127.5
    return tf.cast(imgs, tf.uint8)