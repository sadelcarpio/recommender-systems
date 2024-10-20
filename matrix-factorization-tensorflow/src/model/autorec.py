from keras import layers, Model
import tensorflow as tf


class AutoRecommender(Model):
    def __init__(self, m: int, k: int):
        super().__init__()
        self.user_movies = layers.Input(shape=(m,))
        self.dropout_1 = layers.Dropout(rate=0.3)(self.user_movies)
        self.hidden = layers.Dense(units=k)(self.dropout_1)
        self.reconstructed = layers.Dense(m, activation='relu')(self.hidden)
        self.model = Model(inputs=self.user_movies, outputs=self.reconstructed)

    def call(self, inputs, training=False):
        dense_inputs = tf.sparse.to_dense(inputs)
        return self.model(dense_inputs)

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
