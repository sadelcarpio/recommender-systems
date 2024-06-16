import numpy as np

from algorithm.strategy.base import FitStrategy


class BiasStrategy(FitStrategy):
    b = None
    c = None
    mu = None

    def init_params(self, M, N, K, mu=None, reg=0.0):
        self.K = K
        self.W, self.U = np.random.randn(N, K), np.random.randn(M, K)
        self.b, self.c = np.zeros(N), np.zeros(M)
        self.mu = mu
        self.reg = reg

    def step(self, items_per_user, users_per_item):
        for user_id in items_per_user.index:
            items_rated_user_id = items_per_user.loc[user_id].ratedMovies
            num_items_rated = len(items_rated_user_id)
            items_indices = list(items_rated_user_id.keys())
            u = self.U[items_indices]
            ratings_w = np.array(list(items_rated_user_id.values()))
            c_items = self.c[items_indices]
            A_w = np.dot(u.T, u) + self.reg * np.eye(self.K)
            b_w = np.dot(ratings_w - self.b[user_id] - c_items - self.mu, u)
            self.W[user_id] = np.linalg.solve(A_w, b_w)
            self.b[user_id] = (ratings_w - np.dot(self.W[user_id], u.T) - c_items - self.mu).sum() / (num_items_rated + self.reg)
        for item_id in users_per_item.index:
            users_rated_item_id = users_per_item.loc[item_id].usersRated
            num_users_rated = len(users_rated_item_id)
            users_indices = list(users_rated_item_id.keys())
            w = self.W[users_indices]
            ratings_u = np.array(list(users_rated_item_id.values()))
            b_users = self.b[users_indices]
            A_u = np.dot(w.T, w) + self.reg * np.eye(self.K)
            b_u = np.dot(ratings_u - self.c[item_id] - b_users - self.mu, w)
            self.U[item_id] = np.linalg.solve(A_u, b_u)
            self.c[item_id] = (ratings_u - np.dot(self.U[item_id], w.T) - b_users - self.mu).sum() / (num_users_rated + self.reg)

    def predict(self, row):
        user_id = row["userIdOrdered"]
        item_id = row["movieIdOrdered"]
        return np.dot(self.W[user_id], self.U[item_id]) + self.b[user_id] + self.c[item_id] + self.mu
