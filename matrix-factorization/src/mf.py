import numpy as np
import pandas as pd

from src.loss import mse


class MatrixFactorization:
    def __init__(self, K: int):
        self.K = K
        self.W = None
        self.U = None
        self.M = None
        self.N = None
        self.items_per_user = None
        self.users_per_item = None

    def fit(self, train_dataset, n_iter: int) -> list:
        """
        Fits the model to the training dataset, learning the W (user features) and U (item features) matrices.
        :param train_dataset:
        :param n_iter: Number of iterations for the ALS algorithm
        :return: Loss function on each iteration
        """
        self.M = train_dataset.movieId.nunique()
        self.N = train_dataset.userId.nunique()
        self.items_per_user = train_dataset.groupby("userIdOrdered").apply(
            lambda x: pd.Series({'ratedMovies': dict(zip(x["movieIdOrdered"], x["rating"])),
                                 'meanRating': x["rating"].mean()}), include_groups=False)
        self.users_per_item = train_dataset.groupby("movieIdOrdered").apply(
            lambda x: pd.Series({'usersRated': dict(zip(x["userIdOrdered"], x["rating"])),
                                 'meanRating': x["rating"].mean()}), include_groups=False)
        self.W, self.U = np.random.randn(self.N, self.K), np.random.randn(self.M, self.K)
        losses = []
        for t in range(n_iter):
            print(f"Iteration {t + 1}")
            for user_id in self.items_per_user.index:
                items_rated_user_id = self.items_per_user.loc[user_id].ratedMovies
                u = self.U[list(items_rated_user_id.keys())]
                ratings_w = np.array(list(items_rated_user_id.values()))
                A_w = np.dot(u.T, u)
                b_w = np.dot(ratings_w, u)
                self.W[user_id] = np.linalg.solve(A_w, b_w)
            for item_id in self.users_per_item.index:
                users_rated_item_id = self.users_per_item.loc[item_id].usersRated
                w = self.W[list(users_rated_item_id.keys())]
                ratings_u = np.array(list(users_rated_item_id.values()))
                A_u = np.dot(w.T, w)
                b_u = np.dot(ratings_u, w)
                self.U[item_id] = np.linalg.solve(A_u, b_u)
            # Calculate loss function
            loss = mse(train_dataset["rating"], self.predict(train_dataset))
            losses.append(loss)
            print(f"Loss for iteration {t + 1}: {loss}")
        return losses

    def predict(self, df):
        """
        Calculates predicted score on a dataframe
        :param df: DataFrame with rows: movieIdOrdered, userIdOrdered
        :return: pd.Series with the predicted score for each row of df
        """
        return df.apply(self.predict_score, axis=1).apply(lambda x: min(5, max(0, x)))

    def predict_score(self, row):
        """
        Predicts score for a single row of a dataframe
        :param row: row that contains a movieIdOrdered and userIdOrdered fields
        :return: s(i, j)
        """
        user_id = row["userIdOrdered"]
        item_id = row["movieIdOrdered"]
        return np.dot(self.W[user_id], self.U[item_id].T)
