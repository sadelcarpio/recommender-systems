import tensorflow as tf
from keras import Model, initializers


class BernoulliRBM(Model):
    def __init__(self, hidden_units: int, k: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None
        self.c = None
        if k < 1:
            raise ValueError('k must be greater than zero')
        self.k = k
        self.visible_units = None
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.visible_units = input_shape[-1]
        self.w = self.add_weight(name='w', shape=(self.visible_units, self.hidden_units),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(1, self.visible_units),
                                 initializer=initializers.Zeros(),
                                 trainable=True)
        self.c = self.add_weight(name='c', shape=(1, self.hidden_units),
                                 initializer=initializers.Zeros(),
                                 trainable=True)

    def sample_h(self, v):
        p_h = tf.sigmoid(v @ self.w + self.c)
        h = tf.stop_gradient(tf.cast(tf.random.uniform(tf.shape(p_h)) < p_h, tf.float32))
        return h

    def sample_v(self, h):
        p_v = tf.sigmoid(h @ tf.transpose(self.w) + self.b)
        v = tf.stop_gradient(tf.cast(tf.random.uniform(tf.shape(p_v)) < p_v, tf.float32))
        return v

    def call(self, inputs, training=False, mask=None):
        v_prime = inputs
        for _ in range(self.k):
            h = self.sample_h(v_prime)
            v_prime = self.sample_v(h)
        return v_prime

    def free_energy(self, v):
        return - v @ tf.transpose(self.b) - tf.reduce_sum(tf.math.log(1 + tf.math.exp(v @ self.w + self.c)), axis=1,
                                                          keepdims=True)

    def train_step(self, data):
        v, v = data
        with tf.GradientTape() as tape:
            v_prime = self(v)
            loss = tf.reduce_mean(self.free_energy(v) - self.free_energy(v_prime))
        diff = tf.reduce_mean(v - v_prime)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss, 'v-v_prime': diff}

    def test_step(self, data):
        v, v = data
        v_prime = self(v, training=False)
        diff = tf.reduce_mean(v - v_prime)
        loss = tf.reduce_mean(self.free_energy(v) - self.free_energy(v_prime))
        return {'loss': loss, 'v-v_prime': diff}
