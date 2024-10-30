from keras import optimizers

import data
from model import CategoricalRBM

df = data.Dataset('movielens-20m-dataset/rating.csv', n_most_users=20000, m_most_items=2000)
train_dataset, val_dataset = df.sparse_dataset(test_ratio=0.1, batch_size=32)

model = CategoricalRBM(hidden_units=150, num_classes=10)
model.compile(optimizer=optimizers.SGD(learning_rate=1e-3))
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

generated = model.predict(val_dataset)
