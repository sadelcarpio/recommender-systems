from keras import optimizers

from src.data import Dataset
from src.model.mf import MFModel

dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=80000, m_most_items=30000)
train, test = dataset.split_dataset(0.2)

print(f"Train set size: {len(train)}")
print(f"Test set size: {len(test)}")

mu_train = train["rating"].mean()
mu_test = test["rating"].mean()

N = train.userIdOrdered.nunique()
M = train.movieIdOrdered.nunique()

model = MFModel(k=120, m=M, n=N, reg=3e-3, residual=True)


model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['mse', 'mae'])
print(model.summary())
history = model.fit([train["userIdOrdered"], train["movieIdOrdered"]], train["rating"] - mu_train,
                    validation_data=([test["userIdOrdered"], test["movieIdOrdered"]], test["rating"] - mu_train),
                    batch_size=4096,
                    epochs=50)
