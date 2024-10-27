import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, optimizers

from model import BernoulliRBM
from preprocess import preprocess_mnist

(x_train, _), (x_test, _) = datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).map(preprocess_mnist).batch(1)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).map(preprocess_mnist).batch(1)
model = BernoulliRBM(hidden_units=100, k=1)
model.compile(optimizer=optimizers.SGD(learning_rate=0.001))
model.fit(dataset, validation_data=val_dataset, epochs=1)
reconstructed = model.predict(val_dataset)

plt.figure(figsize=(10, 10))
for i, comp in enumerate(tf.transpose(model.w)):
    plt.subplot(10, 10, i + 1)
    plt.imshow(tf.reshape(comp, (28, 28)), cmap=plt.cm.RdBu)
    plt.xticks([]), plt.yticks([])

plt.figure(figsize=(10, 10))
for i, gen in enumerate(reconstructed[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(tf.reshape(gen, (28, 28)), cmap='gray')
    plt.xticks([]), plt.yticks([])

plt.show()
