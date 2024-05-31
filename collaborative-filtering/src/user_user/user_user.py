import heapq

import numpy as np
import pandas as pd


class UserUserModel:
    def __init__(self, min_common_items: int, max_common_weights: int):
        self.min_common_items = min_common_items
        self.max_common_weights = max_common_weights
        self.items_per_user = None
        self.sparse_weights = None

    def fit(self, train_dataset):
        """
        Calculates the weights (correlation) between users that rated the same items
        :param train_dataset: Dataframe with the columns movieId, userId, rating
        """
        self.sparse_weights = {}
        self.items_per_user = train_dataset.groupby("userId").apply(
            lambda x: pd.Series({'ratedMovies': dict(zip(x["movieId"], x["rating"])),
                                 'meanRating': x["rating"].mean()}), include_groups=False)
        for user_id in self.items_per_user.index:
            self.sparse_weights[user_id] = []
        for user_id, row in self.items_per_user.iterrows():
            rated_items_i = row["ratedMovies"]
            for user_prime_id in reversed(self.items_per_user.index):
                if user_prime_id == user_id:
                    break
                record = self.items_per_user.loc[user_prime_id]
                rated_items_prime = record["ratedMovies"]
                corr_coeff = self.calculate_correlation(rated_items_i, rated_items_prime, self.min_common_items)
                if corr_coeff is None:
                    continue
                self.sparse_weights[user_id].append((user_prime_id, corr_coeff))
                self.sparse_weights[user_prime_id].append((user_id, corr_coeff))

    def predict(self, df):
        """
        Calculates prediction on a DataFrame
        :param df: Dataframe with the columns movieId, userId
        :return: pd.Series with a predicted score for each row of df
        """
        return df.apply(self.predict_score, axis=1).apply(lambda x: min(5, max(0, x)))

    def predict_score(self, row):
        """
        Computes the predicted score for a given (userId, movieId) instance
        s(i, j) = sum(w_ii' * (r_i'j - r_i'_mean)) / sum(abs(w_ii'))
        :param row: that contains the userId and movieId
        :return: s(i, j)
        """
        sum_weights = 0
        weighted_sum = 0
        user_id = row["userId"]
        item_id = row["movieId"]
        common_ratings = []
        if self.sparse_weights.get(user_id) is None:
            return np.nan
        for user_prime_id, weight in self.sparse_weights[user_id]:
            record = self.items_per_user.loc[user_prime_id]
            rating_prime = record["ratedMovies"].get(item_id)
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
            return self.items_per_user.loc[user_id]["meanRating"] + deviation_score
        return np.nan

    @staticmethod
    def calculate_correlation(rated_items_i: dict, rated_items_prime: dict, min_common_items: int = 5):
        intersection = set(rated_items_i).intersection(set(rated_items_prime))
        if len(intersection) > min_common_items:
            user_ratings = pd.Series(rated_items_i)
            user_prime_ratings = pd.Series(rated_items_prime)
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation_coefficient = user_ratings.corr(user_prime_ratings)
            if np.isnan(correlation_coefficient):
                return None
            return correlation_coefficient

    @staticmethod
    def evaluate_mse(rating: pd.Series, predicted: pd.Series) -> float:
        return ((rating - predicted) ** 2).mean()
