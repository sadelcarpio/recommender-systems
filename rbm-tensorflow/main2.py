import tensorflow as tf
from keras import optimizers

from model import CategoricalRBM

num_classes = 4
visible_units = 10
num_samples = 100
X = []
for i in range(num_samples):
    cat = tf.random.categorical(tf.math.log([[1 / num_classes] * num_classes]), visible_units)
    X.append(tf.squeeze(tf.one_hot(cat, num_classes)))
X = tf.stack(X)
print(X.shape)
dataset = tf.data.Dataset.from_tensor_slices((X, X)).batch(1)
model = CategoricalRBM(hidden_units=5, num_classes=num_classes)
model.compile(optimizer=optimizers.SGD(learning_rate=0.01))
model.fit(dataset, epochs=1)
