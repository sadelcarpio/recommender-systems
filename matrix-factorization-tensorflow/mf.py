from keras import layers, Model, regularizers, optimizers

from data import Dataset


def create_model(k: int, m: int, n: int, reg: float = 0.0):
    w = layers.Input(shape=(1,))
    u = layers.Input(shape=(1,))
    b_embedding = layers.Embedding(input_dim=n, output_dim=1, embeddings_regularizer=regularizers.l2(reg))(w)
    w_embedding = layers.Embedding(input_dim=n, output_dim=k, embeddings_regularizer=regularizers.l2(reg))(w)
    u_embedding = layers.Embedding(input_dim=m, output_dim=k, embeddings_regularizer=regularizers.l2(reg))(u)
    c_embedding = layers.Embedding(input_dim=m, output_dim=1, embeddings_regularizer=regularizers.l2(reg))(u)
    mf_layer = layers.Dot(axes=2)([w_embedding, u_embedding])
    bias_layer = layers.Add()([mf_layer, b_embedding, c_embedding])
    model = Model(inputs=[w, u], outputs=bias_layer)
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001))
    return model


dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=2000, m_most_items=200)
train, test = dataset.split_dataset(0.2)
mu_train = train["rating"].mean()
mu_test = test["rating"].mean()

N = train.userIdOrdered.nunique()
M = train.movieIdOrdered.nunique()

model = create_model(k=60, m=M, n=N, reg=3e-4)
history = model.fit([train["userIdOrdered"], train["movieIdOrdered"]], train["rating"] - mu_train,
                    validation_data=([test["userIdOrdered"], test["movieIdOrdered"]], test["rating"] - mu_train),
                    epochs=50)
