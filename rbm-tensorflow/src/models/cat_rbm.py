import tensorflow as tf
from keras import Model, initializers


class CategoricalRBM(Model):
    def __init__(self, hidden_units: int, num_classes: int, k: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None
        self.c = None
        self.num_classes = num_classes
        if k < 1:
            raise ValueError('k must be greater than zero')
        self.k = k
        self.visible_units = None
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.visible_units = input_shape[-2]
        self.w = self.add_weight(name='w', shape=(self.visible_units, self.num_classes, self.hidden_units),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(self.visible_units, self.num_classes),
                                 initializer=initializers.Zeros(),
                                 trainable=True)
        self.c = self.add_weight(name='c', shape=(1, self.hidden_units),
                                 initializer=initializers.Zeros(),
                                 trainable=True)

    def sample_h(self, v):
        linear = tf.tensordot(v, self.w, axes=[[1, 2], [0, 1]]) + self.c
        p_h = tf.sigmoid(linear)
        # h = tf.stop_gradient(tf.cast(tf.random.uniform(tf.shape(p_h)) < p_h, tf.float32))
        return p_h

    def sample_v(self, h):
        linear = tf.tensordot(h, self.w, axes=[[1], [2]]) + self.b
        p_v = tf.math.softmax(linear)
        # p_v_reshaped = tf.stop_gradient(tf.reshape(p_v, [-1, self.num_classes]))
        # sample_indices = tf.stop_gradient(tf.random.categorical(tf.math.log(p_v_reshaped), num_samples=1))
        # sample_indices_reshaped = tf.stop_gradient(tf.reshape(sample_indices, [-1, self.visible_units]))
        # v = tf.stop_gradient(tf.one_hot(sample_indices_reshaped, depth=self.num_classes))
        return p_v

    def call(self, inputs, training=False, mask=None):
        v_prime = inputs
        for _ in range(self.k):
            h = self.sample_h(v_prime)
            v_prime = self.sample_v(h)
        return v_prime

    def predict_step(self, data):
        x, _ = data
        v_dense = tf.sparse.to_dense(x)
        hidden = self.sample_h(v_dense)
        linear = tf.tensordot(hidden, self.w, axes=[[1], [2]]) + self.b
        p_v = tf.math.softmax(linear)
        return p_v

    def free_energy(self, v):
        linear = tf.tensordot(v, self.w, axes=[[1, 2], [0, 1]]) + self.c
        return - tf.tensordot(self.b, v, axes=[[0, 1], [1, 2]]) - tf.reduce_sum(tf.math.log(1 + tf.math.exp(linear)), axis=1)

    def train_step(self, data):
        v, v = data
        dense_v = tf.sparse.to_dense(v)
        with tf.GradientTape() as tape:
            v_prime = self(dense_v)
            mask = tf.reduce_sum(dense_v, axis=-1, keepdims=True) == 0
            v_prime = tf.where(mask, tf.zeros_like(v_prime), v_prime)
            loss = tf.reduce_mean(self.free_energy(dense_v) - self.free_energy(v_prime))
        diff = tf.reduce_mean(tf.abs(dense_v - v_prime))
        # Get labels
        v_labels = tf.argmax(dense_v, axis=2)
        v_prime_labels = tf.argmax(v_prime, axis=2)
        # Get indices of nonzero
        indices = tf.where(tf.reduce_sum(dense_v, axis=-1) != 0)
        # Get nonzero ratings
        v_ratings = 0.5 * tf.cast(tf.gather_nd(v_labels, indices), dtype=tf.float32) + 0.5
        v_prime_ratings = 0.5 * tf.cast(tf.gather_nd(v_prime_labels, indices), dtype=tf.float32) + 0.5
        mae = tf.reduce_mean(tf.abs(v_ratings - v_prime_ratings))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss, 'v-v_prime': diff, 'mae-ratings': mae}

    def test_step(self, data):
        v, v = data
        dense_v = tf.sparse.to_dense(v)
        v_prime = self(dense_v, training=False)
        mask = tf.reduce_sum(dense_v, axis=-1, keepdims=True) == 0
        v_prime = tf.where(mask, tf.zeros_like(v_prime), v_prime)
        # Get labels
        v_labels = tf.argmax(dense_v, axis=2)
        v_prime_labels = tf.argmax(v_prime, axis=2)
        # Get indices of nonzero
        indices = tf.where(tf.reduce_sum(dense_v, axis=-1) != 0)
        # Get nonzero ratings
        v_ratings = 0.5 * tf.cast(tf.gather_nd(v_labels, indices), dtype=tf.float32) + 0.5
        v_prime_ratings = 0.5 * tf.cast(tf.gather_nd(v_prime_labels, indices), dtype=tf.float32) + 0.5
        mae = tf.reduce_mean(tf.abs(v_ratings - v_prime_ratings))
        diff = tf.reduce_mean(tf.abs(dense_v - v_prime))
        loss = tf.reduce_mean(self.free_energy(dense_v) - self.free_energy(v_prime))
        return {'loss': loss, 'v-v_prime': diff, 'mae-ratings': mae}
