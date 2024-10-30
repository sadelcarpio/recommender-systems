import tensorflow as tf


def preprocess_mnist(x, y):
    x, y = x / 255, y / 255
    x, y = tf.reshape(x, (28 * 28,)), tf.reshape(y, (28 * 28,))
    return x, y
