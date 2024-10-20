import tensorflow as tf


def custom_sparse_mse(y_true, y_pred):
    dense_y_true = tf.sparse.to_dense(y_true)
    masked_y_pred = tf.where(dense_y_true != 0, y_pred, 0)
    squared_errors = tf.square(dense_y_true - masked_y_pred)
    nonzero_indices = tf.where(tf.not_equal(dense_y_true, 0))
    masked_squared_errors = tf.gather_nd(squared_errors, nonzero_indices)
    return tf.reduce_mean(masked_squared_errors)
