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
        if not self.built:
            self.visible_units = input_shape[-1]
            self.w = tf.Variable(tf.random.normal((self.visible_units, self.hidden_units)))
            self.b = tf.Variable(tf.random.normal((1, self.visible_units)))
            self.c = tf.Variable(tf.random.normal((1, self.hidden_units)))
            self.built = True

    def call(self, inputs, training=False, mask=None):
        batch_size = 32
        p_h = tf.sigmoid(inputs @ self.w + self.c)
        h = tf.where(tf.random.uniform((batch_size, self.hidden_units)) < p_h, 1, 0)
        p_v = tf.sigmoid(tf.cast(h, tf.float32) @ tf.transpose(self.w) + self.b)
        v_prime = tf.where(tf.random.uniform((batch_size, self.visible_units)) < p_v, 1, 0)
        return tf.cast(v_prime, tf.float32)

    def free_energy(self, v):
        return - v @ tf.transpose(self.b)

    def train_step(self, data):
        v, v = data
        with tf.GradientTape() as tape:
            self.build(K.int_shape(v))
            v_prime = self.call(v)
            loss = self.free_energy(v) - self.free_energy(v_prime)
        return loss


X = tf.random.normal((64, 10))
dataset = tf.data.Dataset.from_tensor_slices((X, X)).batch(32)
model = BernoulliRBM(hidden_units=5)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='SGD')
model.fit(dataset, epochs=1)

x = tf.random.normal((1, 3))
w = tf.Variable(tf.random.normal((3, 2)))
with tf.GradientTape() as tape:
    p = tf.sigmoid(x @ w)
    print(p)
    v = tf.cast(tf.where(tf.random.uniform((1, 2)) < p, 1, 0), tf.float32)
    print(v)
    f_v = tf.exp(v @ tf.transpose(w))

grad = tape.gradient(f_v, w)
print(grad)
