from keras import optimizers
from src.model.autorec import AutoRecommender
from src.data import Dataset
import tensorflow as tf


def custom_sparse_mse(y_true, y_pred):
    dense_y_true = tf.sparse.to_dense(y_true)
    return tf.reduce_mean(tf.square(dense_y_true - y_pred))


dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=10000, m_most_items=1000)
print(f"Size of the dataset: {dataset.dataset.shape[0]}")
indices = list(zip(dataset.dataset['userIdOrdered'], dataset.dataset['movieIdOrdered']))
values = dataset.dataset['rating'].values
sparse_dataset = tf.SparseTensor(indices=indices, values=values, dense_shape=[10000, 1000])
model = AutoRecommender(m=1000, k=200)
model.compile(loss=custom_sparse_mse, optimizer=optimizers.Adam(learning_rate=0.001))
print(model.summary())
model.fit(sparse_dataset, sparse_dataset, epochs=100, batch_size=128)
