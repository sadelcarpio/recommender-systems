import pandas as pd

from algorithm import BiasStrategy
from algorithm.strategy.base import FitStrategy
from loss import mse


class MatrixFactorization:
    def __init__(self, K: int, strategy: FitStrategy = BiasStrategy(), reg: float = 0):
        self.K = K
        self.strategy = strategy
        self.reg = reg
        self.M = None
        self.N = None
        self.items_per_user = None
        self.users_per_item = None

    def fit(self, train_dataset, delta: float = 0.001) -> list:
        """
        Fits the model to the training dataset, learning the W (user features) and U (item features) matrices.
        :param train_dataset:
        :param delta: Minimum difference between subsequent losses to consider the algorithm converged
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
        self.strategy.init_params(self.M, self.N, self.K, mu=train_dataset.rating.mean(), reg=self.reg)
        losses = []
        n_iter = 1
        while True:
            print(f"Iteration {n_iter}")
            self.strategy.step(self.items_per_user, self.users_per_item)
            loss = mse(train_dataset["rating"], self.predict(train_dataset))
            print(f"Loss for iteration {n_iter}: {loss}")
            losses.append(loss)
            if n_iter > 1:
                error = abs(losses[-2] - losses[-1])
                if error < delta:
                    break
            n_iter += 1
        return losses

    def predict(self, df):
        """
        Calculates predicted score on a dataframe
        :param df: DataFrame with rows: movieIdOrdered, userIdOrdered
        :return: pd.Series with the predicted score for each row of df
        """
        return df.apply(self.strategy.predict, axis=1)
