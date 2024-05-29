import numpy as np
import pandas as pd


class UserUserModel:
    def __init__(self, min_common_movies: int):
        self.min_common_movies = min_common_movies
        self.items_per_user_train = None
        self.sparse_weights = None

    def fit(self, train_dataset):
        self.sparse_weights = {}
        self.items_per_user_train = train_dataset.groupby("userId").apply(
            lambda x: pd.Series({'ratedMovies': dict(zip(x["movieId"], x["rating"])),
                                 'meanRating': x["rating"].mean()}), include_groups=False)
        for user_id in self.items_per_user_train.index:
            self.sparse_weights[user_id] = []
        for user_id, row in self.items_per_user_train.iterrows():
            rated_movies_i = row["ratedMovies"]
            for user_prime_id in reversed(self.items_per_user_train.index):
                if user_prime_id == user_id:
                    break
                record = self.items_per_user_train.loc[user_prime_id]
                rated_movies_prime = record["ratedMovies"]
                corr_coeff = self.calculate_correlation(rated_movies_i, rated_movies_prime, self.min_common_movies)
                if corr_coeff is None:
                    continue
                self.sparse_weights[user_id].append((user_prime_id, corr_coeff))
                self.sparse_weights[user_prime_id].append((user_id, corr_coeff))

    def predict(self, df):
        return df.apply(self.predict_score, axis=1).apply(lambda x: min(5, max(0, x)))

    def predict_score(self, row):
        sum_weights = 0
        weighted_sum = 0
        user_id = row["userId"]
        movie_id = row["movieId"]
        if self.sparse_weights.get(user_id) is None:
            return np.nan
        for i, (user_prime_id, weight) in enumerate(self.sparse_weights[user_id]):
            record = self.items_per_user_train.loc[user_prime_id]
            rating_prime = record["ratedMovies"].get(movie_id)
            if rating_prime is not None:
                sum_weights += abs(weight)
                weighted_sum += weight * (rating_prime - record["meanRating"])
        if sum_weights != 0:
            deviation_score = weighted_sum / sum_weights
            return self.items_per_user_train.loc[user_id]["meanRating"] + deviation_score
        else:
            return np.nan

    @staticmethod
    def calculate_correlation(rated_movies_i: dict, rated_movies_prime: dict, min_common_movies: int = 5):
        intersection = set(rated_movies_i).intersection(set(rated_movies_prime))
        if len(intersection) > min_common_movies:
            user_ratings = pd.Series(rated_movies_i)
            user_prime_ratings = pd.Series(rated_movies_prime)
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation_coefficient = user_ratings.corr(user_prime_ratings)
            if np.isnan(correlation_coefficient):
                return None
            return correlation_coefficient

    @staticmethod
    def evaluate_mse(rating: pd.Series, predicted: pd.Series) -> float:
        return ((rating - predicted) ** 2).mean()
