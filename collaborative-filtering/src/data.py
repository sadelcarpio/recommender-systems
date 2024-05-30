import pandas as pd


class Dataset:
    def __init__(self, dataset_csv_path: str, n_most_users: int, m_most_items: int):
        self.dataset = pd.read_csv(dataset_csv_path)
        most_common_users = self.dataset.userId.value_counts().head(n_most_users)
        most_common_movies = self.dataset.movieId.value_counts().head(m_most_items)
        self.dataset = self.dataset[(self.dataset.userId.isin(most_common_users.index)) &
                                    (self.dataset.movieId.isin(most_common_movies.index))]

    def split_dataset(self, test_ratio: float) -> tuple:
        dataset_train = self.dataset.sample(frac=(1 - test_ratio), random_state=42)
        dataset_test = self.dataset.drop(dataset_train.index)
        return dataset_train, dataset_test
