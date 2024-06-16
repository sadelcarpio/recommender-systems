import numpy as np

from algorithm.strategy.base import FitStrategy


class ALSWithoutBias(FitStrategy):

    def init_params(self, M, N, K, mu=None, reg=0.0):
        self.K = K
        self.W, self.U = np.random.randn(N, K), np.random.randn(M, K)
        self.reg = reg

    def step(self, items_per_user, users_per_item):
        for user_id in items_per_user.index:
            items_rated_user_id = items_per_user.loc[user_id].ratedMovies
            u = self.U[list(items_rated_user_id.keys())]
            ratings_w = np.array(list(items_rated_user_id.values()))
            A_w = np.dot(u.T, u) + self.reg * np.eye(self.K)
            b_w = np.dot(ratings_w, u)
            self.W[user_id] = np.linalg.solve(A_w, b_w)
        for item_id in users_per_item.index:
            users_rated_item_id = users_per_item.loc[item_id].usersRated
            w = self.W[list(users_rated_item_id.keys())]
            ratings_u = np.array(list(users_rated_item_id.values()))
            A_u = np.dot(w.T, w) + self.reg * np.eye(self.K)
            b_u = np.dot(ratings_u, w)
            self.U[item_id] = np.linalg.solve(A_u, b_u)

    def predict(self, row):
        user_id = row["userIdOrdered"]
        item_id = row["movieIdOrdered"]
        return np.dot(self.W[user_id], self.U[item_id])
