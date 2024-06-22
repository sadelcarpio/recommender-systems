from data import Dataset

import numpy as np
from keras import layers, Model, regularizers, optimizers

dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=1000, m_most_items=100)
train, test = dataset.split_dataset(0.2)
mu_train = train["rating"].mean()
mu_test = test["rating"].mean()

N = train.userId.nunique()
M = train.movieId.nunique()
K = 20

w = layers.Input(shape=(1,))
u = layers.Input(shape=(1,))
b_embedding = layers.Embedding(input_dim=N, output_dim=1, embeddings_regularizer=regularizers.l2(3e-4))(w)
w_embedding = layers.Embedding(input_dim=N, output_dim=K, embeddings_regularizer=regularizers.l2(3e-4))(w)
u_embedding = layers.Embedding(input_dim=M, output_dim=K, embeddings_regularizer=regularizers.l2(3e-4))(u)
c_embedding = layers.Embedding(input_dim=M, output_dim=1, embeddings_regularizer=regularizers.l2(3e-4))(u)
mf_layer = layers.Dot(axes=2)([w_embedding, u_embedding])
bias_layer = layers.Add()([mf_layer, b_embedding, c_embedding])
model = Model(inputs=[w, u], outputs=bias_layer)

model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001))
model.fit([train["userIdOrdered"], train["movieIdOrdered"]], train["rating"] - mu_train,
          validation_data=([test["userIdOrdered"], test["movieIdOrdered"]], test["rating"] - mu_train),
          epochs=50)
