from keras import layers, Model, regularizers, optimizers


def create_model(k: int, m: int, n: int, reg: float = 0.0):
    """
    Create a MF model with bias and regularization
    :param k: Dimension of embeddings a.k.a number of features per user / item
    :param m: Number of unique items
    :param n: Number of unique users
    :param reg: regularization parameter
    :return: Keras model.
    """
    w = layers.Input(shape=(1,))
    u = layers.Input(shape=(1,))
    # w produces a K-length vector embedding per user
    w_embedding = layers.Embedding(input_dim=n, output_dim=k, embeddings_regularizer=regularizers.l2(reg))(w)
    # b_i is an 1-1 embedding of user inputs, accounting for bias on each user
    b_embedding = layers.Embedding(input_dim=n, output_dim=1, embeddings_regularizer=regularizers.l2(reg))(w)
    # u produces a K-length vector embedding per item
    u_embedding = layers.Embedding(input_dim=m, output_dim=k, embeddings_regularizer=regularizers.l2(reg))(u)
    # c_j is a 1-1 embedding of item inputs, accounting for bias on each item
    c_embedding = layers.Embedding(input_dim=m, output_dim=1, embeddings_regularizer=regularizers.l2(reg))(u)
    # Dot computes dot product along axis 2.
    # Given: w_embedding.shape = [None, 1, K]. w_embedding = [[w1, w2, w3], ...], w1 = [w11, w12, w13]
    # u_embedding.shape = [None, 1, K] = [[u1, u2, u3], ...], u1 = [u11, u12, u13]
    # then mf_layer.shape = [None, 1, 1], mf_layer = [[[dot(w1.T, u1)], [dot(w2.T, u2)], [dot(w3.T, u3)]], ...]
    mf_layer = layers.Dot(axes=2)([w_embedding, u_embedding])
    # Add the mf result with b and c bias
    # dot(w.T, u) + b_i + c_j
    bias_layer = layers.Add()([mf_layer, b_embedding, c_embedding])
    model = Model(inputs=[w, u], outputs=bias_layer)
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001))
    return model
