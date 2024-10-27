from model import CategoricalRBM
import tensorflow as tf

num_classes = 4
visible_units = 10
num_samples = 100
X = []
for i in range(num_samples):
    cat = tf.random.categorical(tf.math.log([[1 / num_classes] * num_classes]), visible_units)
    X.append(tf.squeeze(tf.one_hot(cat, num_classes)))
X = tf.stack(X)
print(X.shape)
model = CategoricalRBM(hidden_units=5, num_classes=num_classes)
model.build(X.shape)
result = model.call(X)
print(result)
