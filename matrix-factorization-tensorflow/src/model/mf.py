from keras import layers, Model, regularizers, models


class MFModel(Model):
    def __init__(self, n: int, m: int, k: int, reg: float = 0.0, deep: bool = False, residual: bool = False):
        super().__init__()
        self.w = layers.Input(shape=(1,))
        self.u = layers.Input(shape=(1,))
        # w produces a K-length vector embedding per user
        self.w_embedding = layers.Embedding(input_dim=n, output_dim=k,
                                            embeddings_regularizer=regularizers.l2(reg))(self.w)
        # u produces a K-length vector embedding per item
        self.u_embedding = layers.Embedding(input_dim=m, output_dim=k,
                                            embeddings_regularizer=regularizers.l2(reg))(self.u)
        if deep or residual:
            self.concat_features = layers.Concatenate()([self.w_embedding, self.u_embedding])
            self.deep_layer = models.Sequential([
                layers.Dense(units=k // 2, kernel_regularizer=regularizers.l2(reg), activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(rate=0.2),
                layers.Dense(units=k // 4, kernel_regularizer=regularizers.l2(reg), activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(rate=0.1),
            ])
            self.deep_final_layer = layers.Dense(units=1, activation='linear')(self.deep_layer(self.concat_features))
        if not deep:
            # b_i is an 1-1 embedding of user inputs, accounting for bias on each user
            self.b_embedding = layers.Embedding(input_dim=n, output_dim=1,
                                                embeddings_regularizer=regularizers.l2(reg))(self.w)
            # c_j is a 1-1 embedding of item inputs, accounting for bias on each item
            self.c_embedding = layers.Embedding(input_dim=m, output_dim=1,
                                                embeddings_regularizer=regularizers.l2(reg))(self.u)
            # Dot computes dot product along axis 2.
            # Given: w_embedding.shape = [None, 1, K]. w_embedding = [[w1, w2, w3], ...], w1 = [w11, w12, w13]
            # u_embedding.shape = [None, 1, K] = [[u1, u2, u3], ...], u1 = [u11, u12, u13]
            # then dot_layer.shape = [None, 1, 1],
            # dot_layer = [[[dot(w1.T, u1)], [dot(w2.T, u2)], [dot(w3.T, u3)]], ...]
            self.dot_layer = layers.Dot(axes=2)([self.w_embedding, self.u_embedding])
            # Add the mf result with b and c bias
            # dot(w.T, u) + b_i + c_j
            self.mf_final_layer = layers.Add()([self.dot_layer, self.b_embedding, self.c_embedding])

        if residual:
            self.add_residual = layers.Add()([self.deep_final_layer, self.mf_final_layer])
            self.model = Model(inputs=[self.w, self.u], outputs=self.add_residual)
        elif deep:
            self.model = Model(inputs=[self.w, self.u], outputs=self.deep_final_layer)
        else:
            self.model = Model(inputs=[self.w, self.u], outputs=self.mf_final_layer)

    def call(self, inputs, training=False):
        return self.model(inputs)

    def summary(
            self,
            line_length=None,
            positions=None,
            print_fn=None,
            expand_nested=False,
            show_trainable=False,
            layer_range=None,
    ):
        return self.model.summary()
