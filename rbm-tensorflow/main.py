import tensorflow as tf
from keras import Model
from keras import backend as K


class BernoulliRBM(Model):
    def __init__(self, hidden_units: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None
        self.c = None
        self.visible_units = None
        self.hidden_units = hidden_units
        self.built = False

    def build(self, input_shape):
        self.visible_units = input_shape[-1]
        self.w = tf.Variable(tf.random.normal((self.visible_units, self.hidden_units)))
        self.b = tf.Variable(tf.random.normal((1, self.visible_units)))
        self.c = tf.Variable(tf.random.normal((1, self.hidden_units)))

    def call(self, inputs, training=False, mask=None):
        p_h = tf.sigmoid(inputs @ self.w + self.c)
        h = tf.cast(tf.random.uniform(tf.shape(p_h)) < p_h, tf.float32)
        p_v = tf.sigmoid(h @ tf.transpose(self.w) + self.b)
        v_prime = tf.cast(tf.random.uniform(tf.shape(p_v)) < p_v, tf.float32)
        return v_prime

    def free_energy(self, v):
        return - v @ tf.transpose(self.b)

    def train_step(self, data):
        v, v = data
        with tf.GradientTape() as tape:
            v_prime = self(v)
            loss = tf.math.reduce_sum(self.free_energy(v) - self.free_energy(v_prime))
        return {'loss': loss}


X = tf.random.normal((64, 10))
dataset = tf.data.Dataset.from_tensor_slices((X, X)).batch(32)
model = BernoulliRBM(hidden_units=5)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='SGD')
model.fit(dataset, epochs=1)
