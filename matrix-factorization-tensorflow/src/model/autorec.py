from keras import layers, Model
import tensorflow as tf


class AutoRecommender(Model):
    def __init__(self, m: int, k: int):
        super().__init__()
        self.user_movies = layers.Input(shape=(m,))
        self.encoder_1 = layers.Dense(k, activation='relu')(self.user_movies)
        self.encoder_2 = layers.Dense(k // 2, activation='relu')(self.encoder_1)
        self.decoder_1 = layers.Dense(k // 2, activation='relu')(self.encoder_2)
        self.decoder_2 = layers.Dense(k, activation='relu')(self.decoder_1)
        self.reconstructed = layers.Dense(m)(self.decoder_2)
        self.model = Model(inputs=self.user_movies, outputs=self.reconstructed)

    def call(self, inputs, training=False):
        dense_inputs = tf.sparse.to_dense(inputs)
        return self.model(inputs)

    def summary(
            self,
            line_length=None,
            positions=None,
            print_fn=None,
            expand_nested=None,
            show_trainable=False,
            layer_range=None
    ):
        return self.model.summary()
