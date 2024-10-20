from keras import optimizers

from model.losses import custom_sparse_mse
from src.data import Dataset
from src.model.autorec import AutoRecommender

n_most_users = 20000
n_most_items = 2000

dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=n_most_users, m_most_items=n_most_items)
train_ds, test_ds = dataset.sparse_dataset(test_ratio=.2, batch_size=128)
print(f"Number of users: {dataset.n_users}")
print(f"Number of items: {dataset.n_items}")
model = AutoRecommender(m=dataset.n_items, k=100)
model.compile(loss=custom_sparse_mse, optimizer=optimizers.Adam(learning_rate=0.0001))
print(model.summary())
model.fit(train_ds, validation_data=test_ds, epochs=100)
