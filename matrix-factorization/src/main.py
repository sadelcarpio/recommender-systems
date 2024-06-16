from src.loss import mse
from src.algorithm import MatrixFactorization
from src.data import Dataset

if __name__ == '__main__':
    movielens_subset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=10000, m_most_items=1000)
    train_ds, test_ds = movielens_subset.split_dataset(test_ratio=0.2)
    print(f"Number of train instances: {len(train_ds)}")
    print(f"Number of test instances: {len(test_ds)}")
    mf_model = MatrixFactorization(K=50, reg=20.0)
    losses = mf_model.fit(train_ds, test_ds)
    train_mse = mse(train_ds['rating'], mf_model.predict(train_ds))
    print(f"MSE on train: {train_mse}")
    test_mse = mse(test_ds['rating'], mf_model.predict(test_ds))
    print(f"MSE on test: {test_mse}")
