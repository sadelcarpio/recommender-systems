from keras import optimizers

from src.data import Dataset
from src.model.mf import MFModel

dataset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=2000, m_most_items=200)
train, test = dataset.split_dataset(0.2)
mu_train = train["rating"].mean()
mu_test = test["rating"].mean()

N = train.userIdOrdered.nunique()
M = train.movieIdOrdered.nunique()

model = MFModel(k=60, m=M, n=N, reg=3e-2, residual=True)


model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001))
print(model.summary())
history = model.fit([train["userIdOrdered"], train["movieIdOrdered"]], train["rating"] - mu_train,
                    validation_data=([test["userIdOrdered"], test["movieIdOrdered"]], test["rating"] - mu_train),
                    batch_size=256,
                    epochs=50)
