import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers

import data
from model import CategoricalRBM

df = data.Dataset('movielens-20m-dataset/rating.csv', n_most_users=20000, m_most_items=2000)
train_dataset, val_dataset = df.sparse_dataset(test_ratio=0.1, batch_size=1)

model = CategoricalRBM(hidden_units=150, num_classes=10)
model.compile(optimizer=optimizers.SGD(learning_rate=3e-5))
model.fit(train_dataset, validation_data=val_dataset, epochs=1)

generated = model.predict(val_dataset)


for i, (x, y) in enumerate(val_dataset.take(1)):
    x_dense = tf.squeeze(tf.sparse.to_dense(x))
    prediction = generated[0]
    indices = tf.where(tf.reduce_sum(x_dense, axis=1) != 0)
    nonzero_preds = prediction[indices]
    nonzero_orig = x_dense.numpy()[indices]
