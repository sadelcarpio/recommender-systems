import tensorflow as tf
from keras import optimizers

from src.data import Dataset
from src.model.autorec import AutoRecommender


def custom_sparse_mse(y_true, y_pred):
    dense_y_true = tf.sparse.to_dense(y_true)
    masked_y_pred = tf.where(dense_y_true != 0, y_pred, 0)
    squared_errors = tf.square(dense_y_true - masked_y_pred)
    nonzero_indices = tf.where(tf.not_equal(dense_y_true, 0))
    masked_squared_errors = tf.gather_nd(squared_errors, nonzero_indices)
    return tf.reduce_mean(masked_squared_errors)


n_users = 20000
n_items = 2000

dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=n_users, m_most_items=n_items)
print(f"Size of the dataset: {dataset.dataset.shape[0]}")
indices = list(zip(dataset.dataset['userIdOrdered'], dataset.dataset['movieIdOrdered']))
values = dataset.dataset['rating'].values
sparse_dataset = tf.SparseTensor(indices=indices, values=values, dense_shape=[n_users, n_items])
sparse_ds = tf.data.Dataset.from_tensor_slices((sparse_dataset, sparse_dataset)).batch(64)
train_size = int(0.8 * sparse_ds.cardinality().numpy())
val_size = sparse_ds.cardinality().numpy() - train_size
sparse_train = sparse_ds.take(train_size)
sparse_test = sparse_ds.skip(train_size).take(val_size)
model = AutoRecommender(m=n_items, k=150)
model.compile(loss=custom_sparse_mse, optimizer=optimizers.Adam(learning_rate=0.0001))
print(model.summary())
model.fit(sparse_train, validation_data=sparse_test, epochs=100)
