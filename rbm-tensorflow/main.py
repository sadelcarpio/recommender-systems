import tensorflow as tf
from keras import Model, datasets, optimizers
import matplotlib.pyplot as plt


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
        self.w = tf.Variable(0.1 * tf.random.uniform((self.visible_units, self.hidden_units), minval=-1, maxval=1))
        self.b = tf.Variable(0.1 * tf.random.uniform((1, self.visible_units), minval=-1, maxval=1))
        self.c = tf.Variable(0.1 * tf.random.uniform((1, self.hidden_units), minval=-1, maxval=1))

    def call(self, inputs, training=False, mask=None):
        p_h = tf.sigmoid(inputs @ self.w + self.c)
        h = tf.cast(tf.random.uniform(tf.shape(p_h)) < p_h, tf.float32)
        p_v = tf.sigmoid(h @ tf.transpose(self.w) + self.b)
        v_prime = tf.cast(tf.random.uniform(tf.shape(p_v)) < p_v, tf.float32)
        return v_prime

    def free_energy(self, v):
        return - v @ tf.transpose(self.b) - tf.reduce_sum(tf.math.log(1 + tf.math.exp(v @ self.w + self.c)), axis=1,
                                                          keepdims=True)

    def train_step(self, data):
        v, v = data
        with tf.GradientTape() as tape:
            v_prime = self(v)
            loss = tf.math.reduce_mean(self.free_energy(v) - self.free_energy(v_prime))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss}


def preprocess_mnist(x, y):
    x, y = x / 255, y / 255
    x, y = tf.reshape(x, (28 * 28,)), tf.reshape(y, (28 * 28,))
    return x, y


(x_train, _), (x_test, _) = datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).map(preprocess_mnist).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).map(preprocess_mnist).batch(32)
model = BernoulliRBM(hidden_units=100)
model.compile(optimizer=optimizers.SGD(learning_rate=0.1))
model.fit(dataset, epochs=10)
reconstructed = model.predict(val_dataset)

plt.figure(figsize=(10, 10))
for i, comp in enumerate(tf.transpose(model.w)):
    plt.subplot(10, 10, i + 1)
    plt.imshow(tf.reshape(comp, (28, 28)), cmap='gray')
    plt.xticks([]), plt.yticks([])

plt.show()
