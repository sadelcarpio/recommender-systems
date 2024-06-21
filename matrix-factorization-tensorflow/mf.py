import numpy as np
from keras import layers, Model, regularizers, optimizers

N = 100
M = 10
K = 5

ratings_range = np.arange(0.0, 5.5, 0.5)

train_w = np.random.choice(N, size=1000, replace=True)
train_u = np.random.choice(M, size=1000, replace=True)
train_r = np.random.choice(ratings_range, size=1000, replace=True)

w = layers.Input(shape=(1,))
u = layers.Input(shape=(1,))
w_embedding = layers.Embedding(input_dim=N, output_dim=K, embeddings_regularizer=regularizers.l2(20.0))(w)
b_embedding = layers.Embedding(input_dim=N, output_dim=1, embeddings_regularizer=regularizers.l2(20.0))(w)
u_embedding = layers.Embedding(input_dim=M, output_dim=K, embeddings_regularizer=regularizers.l2(20.0))(u)
c_embedding = layers.Embedding(input_dim=M, output_dim=1, embeddings_regularizer=regularizers.l2(20.0))(u)
mf_layer = layers.Dot(axes=2)([w_embedding, u_embedding])
bias_layer = layers.Add()([mf_layer, b_embedding, c_embedding])
model = Model(inputs=[w, u], outputs=bias_layer)

model.compile(loss='mse', optimizer=optimizers.SGD(learning_rate=0.001))
model.fit([train_w, train_u], train_r, epochs=100)
