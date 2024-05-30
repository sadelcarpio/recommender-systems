import heapq

import numpy as np
import pandas as pd


class ItemItemModel:
    def __init__(self, min_common_users: int, max_common_weights: int):
        self.min_common_users = min_common_users
        self.max_common_weights = max_common_weights
        self.users_per_item = None
        self.sparse_weights = None

    def fit(self, train_dataset):
        self.sparse_weights = {}
        self.users_per_item = train_dataset.groupby("movieId").apply(
            lambda x: pd.Series({'usersRated': dict(zip(x["userId"], x["rating"])),
                                 'meanRating': x["rating"].mean()}), include_groups=False)
        for item_id in self.users_per_item.index:
            self.sparse_weights[item_id] = []
        for item_id, row in self.users_per_item.iterrows():
            users_rated_i = row["usersRated"]
            for item_prime_id in reversed(self.users_per_item.index):
                if item_prime_id == item_id:
                    break
                record = self.users_per_item.loc[item_prime_id]
                users_rated_prime = record["usersRated"]
                corr_coeff = self.calculate_correlation(users_rated_i, users_rated_prime, self.min_common_users)
                if corr_coeff is None:
                    continue
                self.sparse_weights[item_id].append((item_prime_id, corr_coeff))
                self.sparse_weights[item_prime_id].append((item_id, corr_coeff))

    def predict(self, df):
        return df.apply(self.predict_score, axis=1).apply(lambda x: min(5, max(0, x)))

    def predict_score(self, row):
        sum_weights = 0
        weighted_sum = 0
        user_id = row["userId"]
        item_id = row["movieId"]
        common_ratings = []
        if self.sparse_weights.get(item_id) is None:
            return np.nan
        for item_prime_id, weight in self.sparse_weights[item_id]:
            record = self.users_per_item.loc[item_prime_id]
            rating_prime = record["usersRated"].get(user_id)
            if rating_prime is not None:
                mean_rating = record["meanRating"]
                common_ratings.append((rating_prime, mean_rating, weight))
        print(f"Found {len(common_ratings)} common ratings, defaulting to {self.max_common_weights}")
        best_weights = heapq.nlargest(self.max_common_weights, common_ratings, key=lambda x: x[2])
        for rating_prime, mean_rating, weight in best_weights:
            sum_weights += abs(weight)
            weighted_sum += weight * (rating_prime - mean_rating)
        if sum_weights != 0:
            deviation_score = weighted_sum / sum_weights
            return self.users_per_item.loc[item_id]["meanRating"] + deviation_score
        return np.nan

    @staticmethod
    def calculate_correlation(users_rated_i: dict, users_rated_prime: dict, min_common_items: int = 5):
        intersection = set(users_rated_i).intersection(set(users_rated_prime))
        if len(intersection) > min_common_items:
            user_ratings = pd.Series(users_rated_i)
            user_prime_ratings = pd.Series(users_rated_prime)
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation_coefficient = user_ratings.corr(user_prime_ratings)
            if np.isnan(correlation_coefficient):
                return None
            return correlation_coefficient

    @staticmethod
    def evaluate_mse(rating: pd.Series, predicted: pd.Series) -> float:
        return ((rating - predicted) ** 2).mean()
