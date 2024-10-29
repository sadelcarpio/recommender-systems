from keras import optimizers

import data
from model import CategoricalRBM

df = data.Dataset('movielens-20m-dataset/rating.csv', n_most_users=2000, m_most_items=200)
train_dataset, val_dataset = df.sparse_dataset(test_ratio=0.1, batch_size=1)

model = CategoricalRBM(hidden_units=250, num_classes=10)
model.compile(optimizer=optimizers.SGD(learning_rate=0.01))
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
